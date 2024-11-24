import streamlit as st 
import pandas as pd
from trianModels import trainModels 
from printChart import printChart

st.title('Machine Learning Model Explorer ☁️')

st.sidebar.title('Navigation 🔄')
options = ['Home 🏠', 'Upload Data 📂', 'Select Features 🔧', 'Model Training ⚙️', 'Result 📊']
selection = st.sidebar.radio('Go to', options)

if selection == 'Home 🏠':
    st.write("""
    ### Welcome to the ML Model Explorer!! 🚀
    ## This app allows you to:
    - Upload your dataset 📈
    - Select features and target variable 🎯
    - Choose and train different ML Models 🧑‍💻
    - Evaluate and visualize model performance 📊
    """)

if selection == 'Upload Data 📂':
    st.header('Upload your dataset 📥')
    uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write('### Data Preview 👀')
        st.dataframe(df.head())

        st.session_state['df'] = df

if selection == 'Select Features 🔧':
    st.header('Select features and target variable 🎯')

    if 'df' not in st.session_state:
        st.warning('Please upload a dataset first ⚠️')
    else:
        df = st.session_state['df']
        all_columns = df.select_dtypes(exclude=['category', 'object']).columns.tolist()

        target = st.selectbox('Select target variable 🎯', all_columns)
        features = st.multiselect('Select features 🔍', [col for col in all_columns if col != target])

        if st.button('Confirm selection ✅'):
            if not features:
                st.error('Please select at least one feature ⚠️')
            else:
                st.session_state['features'] = features
                st.session_state['target'] = target
                st.success('Features and target variable selected successfully ✅')

if selection == 'Model Training ⚙️':
    st.header('Train ML Models 🤖')

    if 'df' not in st.session_state or 'features' not in st.session_state or 'target' not in st.session_state:
        st.warning('Please upload and select features first ⚠️')
    else:
        trainModels()

if selection == 'Result 📊':
    st.header('Model performance results 📈')

    if 'trained_model' not in st.session_state:
        st.warning('Please train models first ⚠️')
    else:
        trained_model = st.session_state['trained_model'] 
        X_test = st.session_state['X_test'] 
        y_test = st.session_state['y_test']

        for model_name, model in trained_model.items():
            printChart(model_name, model, X_test, y_test) 
