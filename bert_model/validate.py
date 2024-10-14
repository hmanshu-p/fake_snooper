import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import Dataset
from torch.utils.data import DataLoader
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

# Load the trained model and tokenizer from the directory
model_directory = '/home/hima/trained_model'  # Replace with the path to your saved model directory
model = RobertaForSequenceClassification.from_pretrained(model_directory)
tokenizer = RobertaTokenizer.from_pretrained(model_directory)

# Convert the dataframe to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['Text'], padding="max_length", truncation=True)

dataset = dataset.map(tokenize_function, batched=True)

# Prepare the DataLoader
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
dataloader = DataLoader(dataset, batch_size=16)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 2: Run the Model and Get Predictions
model.eval()
all_predictions = []

with torch.no_grad():
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        all_predictions.extend(predictions.cpu().numpy())

# Add predictions to the dataframe
df['Predicted_Label'] = all_predictions
df['Predicted_Label'] = df['Predicted_Label'].apply(lambda x: 'Real' if x == 0 else 'Fake')

# Save the results to a new CSV file
df.to_csv('/home/hima/predicted_results.csv', index=False)

# Print out some predictions to verify
print(df[['Text', 'label', 'Predicted_Label']].head(10))
