from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras
import pickle
import json
import numpy as np

app = Flask(__name__)
CORS(app)

with open("intents.json") as file:
    data = json.load(file)

@app.post('/predict')

def predict():
    userText = request.get_json().get("message")

    # load trained model
    model = keras.models.load_model('chat_model_college')

    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20

    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([userText]),
                                                                      truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for i in data['intents']:
        if i['tag'] == tag:
            message = {"answer": np.random.choice(i['responses'])}
            return jsonify(message)


if __name__ == "__main__":
    app.run()



