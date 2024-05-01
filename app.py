import yaml
from utils import simple_preprocess
import torch
from flask import Flask, jsonify, request
from predict import predict as text_predict

from model import get_model
app = Flask(__name__)

with open('config.yml', 'rb') as f:
    CONFIG = yaml.load(f, yaml.FullLoader)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL = get_model(**CONFIG['architechture']).to(DEVICE)
MODEL.load_checkpoint(CONFIG['training']['save_path'], DEVICE)
MODEL.eval()

@app.route('/predict', methods=['POST'])
def predict():
    """Predict the sentence sent from post data."""
    text = request.json['text']
    text = simple_preprocess(text)
    result = text_predict(text, MODEL)
    return jsonify(result)

if __name__ == '__main__':
    app.run()