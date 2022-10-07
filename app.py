from distutils import filelist
import pandas as pd
import numpy as np
import time
import streamlit as st
from joblib import parallel_backend, Parallel, delayed , dump , load
import tensorflow as tf
from tensorflow import keras, image
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.xception import preprocess_input as preprocess_input_xception
from keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_InceptionResNetV2
from PIL import Image


@st.cache(allow_output_mutation=True)
def model_load():
    model = load_model('./Models/transfert_InceptionResNetV2_ajust')
    classes = load('./Models/classes_120.joblib')
    return model, classes 

def process_img(img):
    img = img.resize((299,299) ) 
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input_InceptionResNetV2(x)
    return x

def main():
    st.title('Détection de la race d\'un chien')
    file = st.file_uploader('Chargez une image') 

    if file :
        with st.spinner('Chargement de l\'image en cours'): 
            img = Image.open(file)
            st.image(img, width = 124)
            img = process_img(img)

        with st.spinner('Prédiction en cours'):
            model , classes = model_load()
            pred = model.predict(img)
            nom = classes[np.argmax(pred)]
            percent = pred[0][np.argmax(pred)]

        resultat = st.success(f'il s\'agit d\'un {nom} avec une probabilité de {round(percent*100,2)} %' )

if __name__ == '__main__':
    main()
