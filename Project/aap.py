import streamlit as st
import pickle
import numpy as np
import pandas as pd

# import the model
pipe = pickle.load(open(r'\pipe.pkl','rb'))
df = pickle.load(open(r'\df.pkl','rb'))


st.title('Laptop Predictor')


# Brand
Company = st.selectbox('Brand',df['Company'].unique())

# Type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('Ram (in GB)',[8,12,16,24,32,64])

# Weight
weight = st.number_input('Weight of laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['Yes','No'])

# Ips
ips = st.selectbox('IPS',['Yes','No'])

# screen size
screen_size = st.number_input('Screen Size')

# Resolution
resolution = st.selectbox('Screen_Resolution',['1920x1080','1366x768','1600x900','3840x2160',
                                              '3200x1800','2880x1800','2560x1600','2560x1440',
                                              '2304x1440'])

# Cpu
cpu = st.selectbox('CPU',df['Cpu Brand'].unique())

hdd = st.selectbox('HDD (in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD (in GB)',[0,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu Brand'].unique())

os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    
    # Quary
    ppi = None
    if touchscreen == 'Yas':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yas':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size
    quary = np.array([Company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    quary = quary.reshape(1,12)

    st.title("The Predicted price of this -> " + str(int(np.exp(pipe.predict(quary)[0]))))
