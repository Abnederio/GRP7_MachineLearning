import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import pickle


st.title('PREDICT IF DEAD OR ALIVE')
img = Image.open('deadOralive.jpg')
st.image(img,width=800, channels='RGB',caption=None)



model = load_model('heart_failure_clinical_records.keras', compile = True)
history = pickle.load(open('training_history','rb'))

def alive_or_dead():
    age = st.sidebar.slider('Age',1,100)
    anaemia = st.sidebar.selectbox('Anaemia', ('Positive','Negative'))
    creatine_phosphokinase = st.sidebar.slider('Creatnine Phosphokinase',0,8000)
    diabetes = st.sidebar.selectbox('Diabetes', ('Positive','Negative'))
    ejection_fraction = st.sidebar.slider('Ejection Fraction',0,80)
    highblood = st.sidebar.selectbox('High Blood', ('Positive','Negative'))
    platelets = st.sidebar.slider('Platelets',0,850000)
    serum_creatinine = st.sidebar.slider('Serum createnine',0.0,10.0, step=0.1)
    serum_sodium = st.sidebar.slider('Serum sodium',0,150)
    sex = st.sidebar.selectbox('Sex', ('Male','Female'))
    smoking = st.sidebar.selectbox('Smoking', ('Yes','No'))
    
    if anaemia == 'Positive':
        anaemia = 1
    else:
        anaemia = 0
        
    if diabetes == 'Positive':
        diabetes = 1
    else:
        diabetes = 0
    
    if highblood == 'Positive':
        highblood = 1
    else:
        highblood = 0
        
    if sex == 'Male':
        sex = 1
    else:
        sex = 0
    
    if smoking == 'Yes':
        smoking = 1
    else:
        smoking = 0
    

    data = [[age, anaemia, creatine_phosphokinase, diabetes, ejection_fraction, highblood, platelets, serum_creatinine, serum_sodium, sex, smoking]]
    data = tf.constant(data)
    return data

alive_or_not=alive_or_dead()
prediction = model.predict(alive_or_not, steps=1)
pred = [round(x[0]) for x in prediction]

if pred ==[0]:
    st.header('You are Dead')
else:
    st.header('You are Alive')
