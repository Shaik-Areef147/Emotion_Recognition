from flask import Flask, render_template, request
import numpy as np
import librosa
import pickle
import tempfile
import os

from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model
model = load_model("emotion_model.h5")

# Load encoder
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


def extract_features(file_path):
    audio, sr = librosa.load(
        file_path,
        duration=3,
        offset=0.5
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    try:

        if request.method == "POST":

            if "audio" not in request.files:
                return "No audio file uploaded"

            file = request.files["audio"]

            if file.filename == "":
                return "No file selected"

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
                file.save(temp.name)
                temp_path = temp.name

            # Extract features
            features = extract_features(temp_path)

            # Remove temp file
            os.remove(temp_path)

            # Scale features
            features = scaler.transform([features])

            # Predict
            pred = model.predict(features, verbose=0)

            emotion = encoder.inverse_transform(
                [np.argmax(pred)]
            )[0]

            emoji_map = {
                "happy": "😊 Happy",
                "sad": "😢 Sad",
                "angry": "😡 Angry",
                "fearful": "😨 Fearful",
                "neutral": "😐 Neutral",
                "calm": "😌 Calm",
                "disgust": "🤢 Disgust",
                "surprised": "😲 Surprised"
            }

            prediction = emoji_map.get(
                emotion,
                emotion
            )

        return render_template(
            "index.html",
            prediction=prediction
        )

    except Exception as e:
        import traceback
        return f"<pre>{traceback.format_exc()}</pre>"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
