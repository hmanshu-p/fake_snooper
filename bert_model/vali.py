import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import get_scheduler
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
import time
import gc
import evaluate
import re

# Function to clean Text
def clean_Text(Text):
    # Regular expression to find the pattern 'xxxxx (Reuters) - yyyy'
    pattern = re.compile(r'.*? \(Reuters\) - (.*)')
    match = pattern.match(Text)
    if match:
        return match.group(1)  # Return the part after 'yyyy'
    else:
        return Text  # If no pattern is found, return the original Text

# Step 1: Load the CSV File
file_path = '/home/hima/training.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Apply the clean_Text function to the 'Text' column
df['Text'] = df['Text'].apply(clean_Text)

# Print column names to verify
print(df.columns)

# Ensure the correct column names are used
# Convert labels to numerical format (0 for Real, 1 for Fake)
df['label'] = df['label'].apply(lambda x: 0 if x.lower() == 'real' else 1)

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Split the dataset into train, validation, and test sets (20% train, 20% validate, 60% test)
train_val_test_split = dataset.train_test_split(test_size=0.6)
test_dataset = train_val_test_split['test']
train_val_split = train_val_test_split['train'].train_test_split(test_size=0.5)
train_dataset = train_val_split['train']
val_dataset = train_val_split['test']

# Print the number of samples in each set
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

# Step 2: Tokenize the Dataset
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize_function(examples):
    return tokenizer(examples['Text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Step 3: Prepare the DataLoader
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Dynamically find the optimal batch size
initial_batch_size = 8
max_batch_size = 64
batch_size = initial_batch_size

while batch_size <= max_batch_size:
    try:
        print(f"Trying batch size: {batch_size}")
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        # Step 4: Define the Model
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

        # Step 5: Train the Model
        optimizer = AdamW(model.parameters(), lr=5e-5)

        num_epochs = 3
        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model.to(device)

        progress_bar = tqdm(range(num_training_steps))

        scaler = torch.cuda.amp.GradScaler()  # Mixed precision training

        training_start_time = time.time()
        losses = []

        model.train()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.cuda.amp.autocast():  # Mixed precision training
                    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['label'])
                    loss = outputs.loss
                scaler.scale(loss).backward()  # Ensure loss is a scalar

                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                
                losses.append(loss.item())

        training_end_time = time.time()
        training_time = training_end_time - training_start_time

        # Step 6: Evaluate the Model
        metric_accuracy = evaluate.load("accuracy")
        metric_precision = evaluate.load("precision")
        metric_recall = evaluate.load("recall")
        metric_f1 = evaluate.load("f1")

        model.eval()
        all_predictions = []
        all_references = []

        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['label'])

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric_accuracy.add_batch(predictions=predictions, references=batch["label"])
            metric_precision.add_batch(predictions=predictions, references=batch["label"])
            metric_recall.add_batch(predictions=predictions, references=batch["label"])
            metric_f1.add_batch(predictions=predictions, references=batch["label"])
            
            all_predictions.extend(predictions.cpu().numpy())
            all_references.extend(batch["label"].cpu().numpy())

        accuracy = metric_accuracy.compute()
        precision = metric_precision.compute()
        recall = metric_recall.compute()
        f1 = metric_f1.compute()

        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Accuracy: {accuracy['accuracy']:.4f}")
        print(f"Precision: {precision['precision']:.4f}")
        print(f"Recall: {recall['recall']:.4f}")
        print(f"F1-Score: {f1['f1']:.4f}")

        # Step 7: Save the Model
        model.save_pretrained('trained_model')
        tokenizer.save_pretrained('trained_model')

        print("Model and tokenizer saved successfully!")

        break

    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print(f"Batch size {batch_size} is too large, reducing batch size.")
            batch_size //= 2
            torch.cuda.empty_cache()
            gc.collect()
        else:
            raise e

# Save training loss graph
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.legend()
plt.savefig('training_loss.png')
