import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import (r2_score, mean_squared_error)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def printChart(model_name, model, X_test, y_test):
    st.subheader(f'{model_name} 📊')

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    st.write(f'**R2 Score 🏆:** {r2:.2f}')
    st.write(f'**Mean Squared Error 💥:** {mse:.2f}')
    st.write(f'**Root Mean Squared Error 🔴:** {rmse:.2f}')

    residuals = y_test - y_pred

    fig3, ax3 = plt.subplots()
    sns.scatterplot(x=y_pred, y=residuals, ax=ax3)
    ax3.axhline(0, color='red', linestyle='--')  # Add horizontal line at y=0
    ax3.set_xlabel('Predicted values')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residuals Plot')  # Corrected typo here
    st.pyplot(fig3)  # Display the plot using Streamlit
