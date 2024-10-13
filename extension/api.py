from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import time
import psutil
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model (this should be done once when the server starts)
model_dir = r"C:\Users\piseh\OneDrive\Desktop\trained_model"
try:
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model loaded successfully. Using device: {device}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.json['text']
        
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        label = "Real" if predictions.item() == 0 else "Fake"
        
        logger.info(f"Prediction made: {label}")
        return jsonify({"label": label})
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
