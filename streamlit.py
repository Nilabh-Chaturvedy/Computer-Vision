import streamlit as st
import tensorflow as tf
best_model = tf.keras.models.load_model("C:/Users/nilch/Desktop/Computer Vision/X-ray images -Covid Prediction/Computer-Vision/best_hyperopt_model.h5")

import numpy as np
from PIL import Image

# Define class labels

CLASS_NAMES=['Covid','Normal','Viral Pneumonia']

st.title("Covid-19 Detection using X-ray images")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","bmp","jpeg"])

if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    image=image.resize((224,224))
    image=np.array(image)
    image=image/255.0
    image=np.expand_dims(image,axis=0)

    #Predicting the class of the image
    prediction=best_model.predict(image)
    predicted_class=CLASS_NAMES[np.argmax(prediction)]
    st.write("The predicted class of the image is:",predicted_class)