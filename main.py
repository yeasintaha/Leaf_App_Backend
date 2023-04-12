# 1. Library imports
import uvicorn
from fastapi import FastAPI
import io
import os
import pickle
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from firebase_admin import credentials, initialize_app
from firebase_admin import storage 
# from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import pydub
from pydub import AudioSegment
import speech_recognition as sr

# app = FastAPI()

# origins = ["http://10.103.12.214:19001", "*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=['http://localhost:3000/', "http://localhost:3000/home", 'https://leaf-disease-web-app.vercel.app/', 'https://leaf-disease-web-app.vercel.app/home', '*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*']
    )
]

app = FastAPI(middleware=middleware)


# Initialize the Firebase app
cred = credentials.Certificate('leaf_app_firebase.json')
firebase_app = initialize_app(cred, {
    'storageBucket': 'leaf-app-8225f.appspot.com'
})


### Load model 
from tensorflow.keras.models import load_model
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import pandas as pd 
from googletrans import Translator
from scipy.spatial import distance




model = load_model('model_densenet121.h5', compile=False)
nlp_model = SentenceTransformer('distilbert-base-nli-mean-tokens')


leaf_classes = np.array(["Bacterialblight", "Blast", "Brownspot", "Tungro"])


@app.get("/detect-voice/{voice_clip}")
def detect_image(voice_clip:str):
    full_path = f"Images/{voice_clip}"
    bucket = storage.bucket(app=firebase_app)
    blob = bucket.blob(full_path)
    contents = blob.download_to_filename("voice_clip.mp3")
    sound = pydub.AudioSegment.from_mp3("voice_clip.mp3")
    sound.export("voice_clip.wav", format="wav")
    r = sr.Recognizer()

    with sr.AudioFile("voice_clip.wav") as source: 
        print("File is being analised") 
        audio = r.listen(source, phrase_time_limit=100000, ) 
    try: 
        text = r.recognize_google(audio, language='bn-BD')
        return {"text" : text}
    except:
        return {"text": "-1"}


@app.get("/detect-image/{image_name}")
def detect_image(image_name:str):
    full_path = f"Images/{image_name}"
    bucket = storage.bucket(app=firebase_app)
    blob = bucket.blob(full_path)
    # print(blob.generate_signed_url(datetime.timedelta(seconds=1000), method='GET'))
    contents = blob.download_as_bytes()

    # Open the image as a PIL Image object
    img = Image.open(io.BytesIO(contents))

    # Preprocess the image for input to the model
    img = np.array(img)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32')/ 255.0 
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    return  [f'{leaf_classes[np.argmax(prediction)]}' , 
             f'{sorted(prediction)[::-1][0]}']


@app.get("/control-measures/{disease_name}") 
def get_control_measures(disease_name : str): 
    id = np.where(leaf_classes == disease_name) 
    df = pd.read_excel('disease.xlsx', sheet_name='Sheet1')

    bacterialblight_control_measures = df['Bacterialblight_control_measures'].dropna().to_list()
    blast_control_measures = df['Blast_control_measures'].dropna().to_list()
    brownspot_control_measures = df['Brownspot_control_measures'].dropna().to_list()
    tungro_control_measures = df['Tungro_control_measures'].dropna().to_list()
    control_measures = [bacterialblight_control_measures, blast_control_measures, brownspot_control_measures, tungro_control_measures]
    # control_measures = np.array(control_measures)
    return control_measures[int(id[0])]


@app.get("/detect-symptomps/{description}")
def detect_symptoms(description : str): 
    df = pd.read_excel('disease.xlsx', sheet_name='Sheet1')
    bacterial_disease = " ".join(df['Bacterialblight'].dropna().to_list()) 
    blast_disease = " ".join(df['Blast'].dropna().to_list())
    brownspot_disease = " ".join(df['Brownspot'].dropna().to_list())
    tungro_disease = " ".join(df['Tungro'].dropna().to_list())

   
    translator = Translator()
    # model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    result1 = translator.translate(description, src='bn', dest='en')
    description = result1.text
    diseases = [bacterial_disease, blast_disease, brownspot_disease, tungro_disease]
    diseases.append(description)
    sentence_embeddings = nlp_model.encode(diseases)

    symptoms = [1 - distance.cosine(sentence_embeddings[4], sentence_embeddings[0]), 
        1 - distance.cosine(sentence_embeddings[4], sentence_embeddings[1]), 
        1 - distance.cosine(sentence_embeddings[4], sentence_embeddings[2]),
        1 - distance.cosine(sentence_embeddings[4], sentence_embeddings[3])]

    seq = np.array(symptoms).argsort()[-4:][::-1] 
    leaf_classes[seq[0]], leaf_classes[seq[1]]

    num = 2
    symptoms_seq = [symptoms[s] for s in seq]
    if (symptoms_seq[0] - symptoms_seq[1] > 0.12) : 
        num = 1
    elif (symptoms_seq[0]- symptoms_seq[3] <0.04 ): 
        num = 4
    elif (symptoms_seq[0]-symptoms_seq[2] <0.08 ):
        num = 3 

    return {
        'sim': [leaf_classes[s] for s in seq][:num],
        'res': symptoms,
        'all_seq': symptoms_seq
        }
        

@app.get('/')
def index():
    return {"message" : "welcome body"}

@app.post('/todo')
def get_something():
    return {'message': 'welcome'}




if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn main:app --reload
