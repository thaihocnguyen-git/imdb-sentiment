from flask import Flask, jsonify, request
import yaml
from predict import predict as text_predict

from model import get_model
app = Flask(__name__)

with open('config.yml', 'rb') as f:
    CONFIG = yaml.load(f, yaml.FullLoader)

MODEL = get_model(**CONFIG['architechture'])
MODEL.load_checkpoint(CONFIG['training']['save_path'])
MODEL.eval()

@app.route('/predict', methods=['POST'])
def predict():
    """Predict the sentence sent from post data."""
    text = request.json['text']
    result = text_predict(text, MODEL)
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run()