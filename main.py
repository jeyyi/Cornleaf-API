from typing import Union
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

#load the pretrained model
model = load_model('Models/Inception.h5')
labels = ['Gray Leaf Spot', 'Common Rust', 'Healthy', 'Blight']

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    #preprocess the image
    img = image.load_img(file.file, target_size=(180,180))
    x = image.img_to_array(img)
    x = x/255.0
    x = tf.expand_dims(x, axis= 0)

    #perform the classification
    prediction = model.predict(x)
    predicted_class_index = tf.argmax(prediction, axis=1)[0]
    predicted_class_label = labels[predicted_class_label]

    return {"predicted_class": predicted_class_label}


