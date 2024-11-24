import streamlit as st
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


#-----
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
#-----


def trainModels():
    df = st.session_state['df']
    features = st.session_state['features']
    target = st.session_state['target']

    X = df[features]
    y = df[target]

    test_size = st.slider('Test size (%) 📊', 10, 50, 20)
    random_state = st.number_input('Random state 🎲', value=42, step=1)

    # Handling missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    models = {
        'Logistic Regression 📉': LogisticRegression(),
        'Random Forest Regressor 🌲': RandomForestRegressor(),
        'Support Vector Regressor 🚀': SVR(),
        'k-Nearest Neighbors Regressor 🤖': KNeighborsRegressor(),
        'Ridge Regression 🏔️': Ridge(),
        'Lasso Regression 🔗': Lasso(),
        'Decision Tree Regressor 🌳': DecisionTreeRegressor(),
        'Gradient Boosting Regressor ⚡': GradientBoostingRegressor(),
        'AdaBoost Regressor 🌟': AdaBoostRegressor(),
        'Gaussian Process Regressor 🔍': GaussianProcessRegressor(),
        'ElasticNet 🔗': ElasticNet(),
        'MLP Regressor 🧠': MLPRegressor(),
        'Polynomial Regression 🔢': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),  # You can adjust the degree if needed
        'Kernel Ridge Regression 🏞️': KernelRidge()
    }

    selected_models = st.multiselect('Select models to train ⚙️', list(models.keys()), default=list(models.keys()))

    if st.button('Train models 🚀'):
        if not selected_models:
            st.error('Please select at least one model to train ⚠️')
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=test_size/100, random_state=random_state)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)

            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['scaler'] = scaler

            st.success('Data was split into training and testing sets ✅')

            trained_model = {}
            for model_name in selected_models:
                model = models[model_name]
                model.fit(X_train, y_train)
                trained_model[model_name] = model

            st.session_state['trained_model'] = trained_model
            st.success('All models have been trained 🎉')
