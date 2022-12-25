# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 05:46:55 2022

@author: Aondona Moses
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st

loaded_model = pickle.load(open("trained_model.sav", 'rb'))

tab_title = ["Prediction Tab",
             "Visualization Tab"]
tabs = st.tabs(tab_title)

st.sidebar.write("""
         # Iris Flower Prediction Web APP
         
         This APP predicts the Iris Flower Type
         """)
         
st.sidebar.header("User Input Features")


def iris_prediction(input_data):
    
    input_data_as_array = np.asarray(input_data)
    input_data_reshape = input_data_as_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshape)
    
    return prediction
    

uploaded_file = st.sidebar.file_uploader("Upload a CSV File", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        sepal_length_cm = st.sidebar.slider("Sepal_Length", 4.3, 7.9, 5.4)
        sepal_width_cm = st.sidebar.slider("Sepal Width", 2.0, 4.4, 3.4)
        petal_length_cm = st.sidebar.slider("Petal Length", 1.0, 6.9, 1.3)
        petal_width_cm = st.sidebar.slider("Petal Width", 0.1, 2.5, 0.2)
        
        data = {'Sepal_length_cm': sepal_length_cm,
                'Sepal_width_cm': sepal_width_cm,
                'petal_length_cm': petal_length_cm,
                'petal_width_cm': petal_width_cm}
        
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()
    

iris_data= pd.read_csv("Iris.csv")

    

with tabs[0]:
    st.subheader("User Input Features")
    if uploaded_file is not None:
        st.write(input_df)
    else:
        st.write("Waiting for a csv upload")
        st.write(input_df)
    
    prediction = iris_prediction(input_df)

    st.subheader("Prediction")
    iris_species = np.array( ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    st.write(iris_species[prediction])
    
with tabs[1]:
    features_selected = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    
    st.header("Select to create a box plot for defferent features")
    selection = st.selectbox("Select a feature", features_selected)
    st.write("VS")
    st.write("Species")
    
    
    if st.button("Create boxplot"):
        fig = plt.figure(figsize=(5, 4))
        sns.boxplot(x='Species', y=selection, data=iris_data)
        st.pyplot(plt.gcf())
        
    if st.button("Create pairplot"):
        fig = plt.figure(figsize=(7, 5))
        sns.pairplot(iris_data, hue="Species")
        plt.title("Pair Plot")
        st.pyplot(plt.gcf())
    
    
    
    


