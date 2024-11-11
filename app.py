import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

class_names = ['Bears', 'Pandas']
img_size = 180

st.header('Image Classification (Panda or Bear)')
uploaded_file = st.file_uploader('Choose a file...', type=['jpeg', 'jpg', 'png'])

if uploaded_file is not None:
    # Memuat model
    model = load_model('pandaBears.keras')

    # Preprocess gambar
    image = tf.keras.utils.load_img(uploaded_file, target_size=(img_size, img_size))
    image_arr = tf.keras.utils.img_to_array(image)
    image_bat = tf.expand_dims(image_arr, 0)

    # Melakukan prediksi
    prediction = model.predict(image_bat)[0][0]  # Mendapatkan probabilitas kelas 1

    # Menentukan kelas berdasarkan threshold 0.5
    predicted_class = class_names[1] if prediction > 0.5 else class_names[0]
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100

    # Menampilkan gambar dan hasil prediksi
    st.image(image)
    st.write('Pandas/Bear in image is {} with confidence of {:0.2f}%'.format(predicted_class, confidence))
