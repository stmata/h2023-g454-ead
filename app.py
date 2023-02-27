# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Import EDA
    from eda import eda_function
    # Import predictive modelling
    from ml import ml_fucntion
    # Load dataset
    dataset = load_iris()
    # Input
    X = dataset.data
    #print(data)
    # Output
    Y = dataset.target
    #print(output)
    # Output names
    target_names = dataset.target_names
    #print(target_names)
    # Columns / Attribute names
    feature_names = dataset.feature_names
    #print(feautre_name)
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)

    # Streamlit
    # Set up App
    st.set_page_config(page_title="EDA and ML Dashboard",
                    layout="centered",
                    initial_sidebar_state="auto")
    # Add title
    st.title("EDA and Predictive Modelling")
    # Define sidebar and sidebar options
    options = ["EDA", "Predictive Modelling"]
    selected_option = st.sidebar.selectbox("Select an option", options)

    # do EDA
    if selected_option == "EDA":
        # Do
        st.subheader("Exploratory Data Analysis and Visualization")
        st.write("Choose a plot type from the option below:")
        # Call eda function
        eda_function(df, target_names, feature_names, Y)
        
    elif selected_option == "Predictive Modelling":
        # Do
        st.subheader("Predictive Modelling")
        st.write("Choose a transform type and model from the option below:")
        ml_fucntion(X, Y)

if __name__ == "__main__":
    main()