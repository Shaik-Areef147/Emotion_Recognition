from flask import Flask, render_template, request
import librosa
import numpy as np
import pickle

from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("emotion_model.h5")

with open("label_encoder.pkl","rb") as f:
    encoder = pickle.load(f)

with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)

def extract_features(file_path):

    audio,sr = librosa.load(
        file_path,
        duration=3,
        offset=0.5
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mfcc = np.mean(
        mfcc.T,
        axis=0
    )

    return mfcc

@app.route("/",methods=["GET","POST"])

def home():

    prediction=""

    if request.method=="POST":

        file=request.files["audio"]

        filepath="temp.wav"

        file.save(filepath)

        features=extract_features(filepath)

        features=scaler.transform(
            [features]
        )

        pred=model.predict(features)

        emotion=encoder.inverse_transform(
            [np.argmax(pred)]
        )[0]

        emoji_map={

            "happy":"😊 Happy",

            "sad":"😢 Sad",

            "angry":"😡 Angry",

            "fearful":"😨 Fearful",

            "neutral":"😐 Neutral",

            "calm":"😌 Calm",

            "disgust":"🤢 Disgust",

            "surprised":"😲 Surprised"
        }

        prediction=emoji_map.get(
            emotion,
            emotion
        )

    return render_template(
        "index.html",
        prediction=prediction
    )

if __name__=="__main__":
    app.run(debug=True)