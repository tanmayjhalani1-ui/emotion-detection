import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

st.set_page_config(page_title="Emotion Detector")

st.title("Facial Emotion Detection")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@st.cache_resource
def train_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(7, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

if "ready" not in st.session_state:
    st.session_state.ready = False

if st.button("Initialize Model"):
    model = train_model()
    st.session_state.model = model
    st.session_state.ready = True
    st.success("Model ready")

if st.session_state.ready:

    uploaded_file = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded Image", width=200)

        image = image.resize((48,48))
        img_array = img_to_array(image)
        img_array = img_array / 255.0
        img_array = np.reshape(img_array, (1,48,48,1))

        prediction = st.session_state.model.predict(img_array)
        predicted_class = np.argmax(prediction)

        st.subheader(f"Predicted Emotion: {emotion_labels[predicted_class]}")

else:
    st.info("Initialize model first")