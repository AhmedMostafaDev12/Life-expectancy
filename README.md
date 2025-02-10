# Life Expectancy 

This repository contains a machine learning model and a web application for predicting life expectancy based on various health, economic, and social factors. The project includes data analysis, feature engineering, model development, and a deployed web application.

## Project Structure

```
|-- IEEE.csv                  # Dataset used for training and analysis
|-- README.md                 # Project documentation
|-- custom_transformers.py    # Custom feature transformation functions
|-- documentation.pdf         # Detailed project documentation for data cleaning , preprocessing, modeling and evaluation 
|-- life_expectancy.ipynb     # Jupyter Notebook for data analysis and model development
|-- life_expectancy.sav       # Saved machine learning model
|-- page.py                   # Main file for deployment
|-- requirements.txt          # Required dependencies for running the project
|-- streamlit_1.py            # Streamlit script for the web interface
```

## Web Application

The model is deployed as an interactive web application using Streamlit. The live deployment can be accessed here:

### [Live Website: Life Expectancy Prediction App](https://life-expectancy-nycmree5t5vhxrxwamsc9h.streamlit.app/)

### Features:
The web application consists of two main sections:

1. **Prediction Section:**
   - Users can input relevant health and economic indicators.
   - The model provides a real-time life expectancy prediction based on the inputs.

2. **Data Exploration Section:**
   - Users can explore the dataset through visualizations.
   - Insights about the correlation between features and life expectancy are provided.

This project demonstrates the integration of machine learning and interactive web applications to provide real-time insights into life expectancy predictions.

