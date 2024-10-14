import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import time
import psutil

# Function to get current memory usage
def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Memory usage in MB

# Load the Tokenizer and Model
model_dir = r"C:\Users\piseh\OneDrive\Desktop\trained_model"

start_load_time = time.time()
tokenizer = RobertaTokenizer.from_pretrained(model_dir)
model = RobertaForSequenceClassification.from_pretrained(model_dir)
end_load_time = time.time()

load_time = end_load_time - start_load_time

# Function to predict if the text is real or fake
def predict_text(text):
    start_total_time = time.time()
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

    # Move inputs to the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start_eval_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    end_eval_time = time.time()

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    end_total_time = time.time()

    eval_time = end_eval_time - start_eval_time
    total_time = end_total_time - start_total_time

    label = predictions.item()
    return {
        "label": "Real" if label == 0 else "Fake",
        "eval_time": eval_time,
        "total_time": total_time,
        "memory_usage": get_memory_usage()
    }

# Test the function with a sample text
sample_text = "A plane with 62 people aboard crashed in a fiery wreck in a residential area of a city in Brazil's Sao Paulo state Friday, the airline said, with local officials reporting that everyone on board the plane had been killed. The airline Voepass confirmed in a statement that a plane headed for Sao Paulo's international airport Guarulhos crashed in the city of Vinhedo with 58 passengers and four crew members aboard. The statement didn't say what caused the accident. City officials at Valinhos, near Vinhedo, said there were no survivors. Only one home in the local condominium complex had been damaged, while none of the residents were hurt. At an event in southern Brazil, President Luiz Inácio Lula da Silva asked the crowd to stand and observe a minute of silence as he shared the news. He said that it appeared that all passengers and crew aboard had died, without elaborating as to how that information had been obtained."
result = predict_text(sample_text)

# Print results
print(f"The text is classified as: {result['label']}")
print(f"Model load time: {load_time:.2f} seconds")
print(f"Evaluation time: {result['eval_time']:.2f} seconds")
print(f"Total time: {result['total_time']:.2f} seconds")
print(f"Memory usage: {result['memory_usage']:.2f} MB")
