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

app = FastAPI()

# origins = ["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# middleware = [
#     Middleware(
#         CORSMiddleware,
#         allow_origins=['*'],
#         allow_credentials=True,
#         allow_methods=['*'],
#         allow_headers=['*']
#     )
# ]

# app = FastAPI(middleware=middleware)


# Initialize the Firebase app
cred = credentials.Certificate('/Leaf_Fastapi/leaf_app_firebase.json')
firebase_app = initialize_app(cred, {
    'storageBucket': 'leaf-app-8225f.appspot.com'
})

from tensorflow.keras.models import load_model
import tensorflow as tf

model = load_model('model.h5', compile=False)

leaf_classes = np.array(["Brownspot", "Blast", "Brownspot", "Tungro"])

def preprocess_image(image):
    # Resize the image to the required size
    image = image.resize((128, 128))

    # Convert the image to a numpy array
    image = np.array(image)

    # Normalize the image to have values between 0 and 1
    image = image.astype('float32') / 255.0

    # Add a batch dimension to the image
    image = np.expand_dims(image, axis=0)

    return image

@app.post("/detect-image")
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

    return {'Detected': f'Disease {leaf_classes[np.argmax(prediction)]}'}



@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.post('/todo')
def get_something():
    return {'message': 'welcome'}




# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn main:app --reload
