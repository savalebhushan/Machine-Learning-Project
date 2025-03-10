import streamlit as st
import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

with open('model_iris.pkl', 'rb') as file:
            model = pickle.load(file)

def main():
    st.title("Iris Flower Classification")
    st.write("Enter the flower measurements to predict its species")

    # Input fields
    sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
    sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
    petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
    petal_width = st.slider("Petal Width", 0.1, 2.5, 1.0)

    # Prediction
    if st.button("Predict"):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)[0]
        st.success(f"The predicted species is: **{prediction.capitalize()}**")

if __name__ == "__main__":
    main()