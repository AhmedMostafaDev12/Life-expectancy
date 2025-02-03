import pickle 
import numpy as np 
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import streamlit as st
from custom_transformers import handle_outliers
from custom_transformers import label_encode_and_scale
from custom_transformers import fillna_median
from custom_transformers import square



loaded_model = pickle.load(open('life_expectancy.sav','rb'))

model = loaded_model['pipeline']

new_data = pd.DataFrame({
    'Survey_Year': [2006],
    'Mortality_Adults': [123],
    'Infant_Deaths_Count': [8],
    'Alcohol_Consumption_Rate': [.97],
    'Hepatitis_B_Vaccination_Coverage': [83],
    'Measles_Infection_Count': [517],
    'Body_Mass_Index_Avg': [48.5],
    'Polio_Vaccination_Coverage': [83],
    'Total_Health_Expenditure': [3.78],
    'Diphtheria_Vaccination_Coverage': [8],
    'HIV_AIDS_Prevalence_Rate': [0.1],
    'Gross_Domestic_Product': [1762.24617],
    'Total_Population': [18914977],
    'Thinness': [6.4],
    'Nation': ['Syrian Arab Republic'],  
    'Country_Category': ['Developing']
})


def prediction(new):
    y_pred = model.predict(new)
    return y_pred




def main():
    # Title of the app
    st.title("Country Life Expectancy Prediction")

    # Input fields for each feature
    Survey_Year = st.number_input("Survey Year")
    Mortality_Adults = st.number_input("Adult Mortality Rate")
    Infant_Deaths_Count = st.number_input("Infant Deaths Count")
    Alcohol_Consumption_Rate = st.number_input("Alcohol Consumption Rate (liters per capita)")
    Hepatitis_B_Vaccination_Coverage = st.number_input("Hepatitis B Vaccination Coverage (%)")
    Measles_Infection_Count = st.number_input("Measles Infection Count")
    Body_Mass_Index_Avg = st.number_input("Average Body Mass Index (BMI)")
    Polio_Vaccination_Coverage = st.number_input("Polio Vaccination Coverage (%)")
    Total_Health_Expenditure = st.number_input("Total Health Expenditure (% of GDP)")
    Diphtheria_Vaccination_Coverage = st.number_input("Diphtheria Vaccination Coverage (%)")
    HIV_AIDS_Prevalence_Rate = st.number_input("HIV/AIDS Prevalence Rate (%)")
    Gross_Domestic_Product = st.number_input("Gross Domestic Product (per capita, USD)")
    Total_Population = st.number_input("Total Population")
    Thinness = st.number_input("Thinness Rate (%)")
    Nation = st.text_input("Nation", value="Syrian Arab Republic")
    Country_Category = st.selectbox("Country Category", ["Developed", "Developing"])

    # Predict button
    if st.button("Predict Life Expectancy"):
        # Create DataFrame for the input data
        new_data = pd.DataFrame({
            'Survey_Year': [Survey_Year],
            'Mortality_Adults': [Mortality_Adults],
            'Infant_Deaths_Count': [Infant_Deaths_Count],
            'Alcohol_Consumption_Rate': [Alcohol_Consumption_Rate],
            'Hepatitis_B_Vaccination_Coverage': [Hepatitis_B_Vaccination_Coverage],
            'Measles_Infection_Count': [Measles_Infection_Count],
            'Body_Mass_Index_Avg': [Body_Mass_Index_Avg],
            'Polio_Vaccination_Coverage': [Polio_Vaccination_Coverage],
            'Total_Health_Expenditure': [Total_Health_Expenditure],
            'Diphtheria_Vaccination_Coverage': [Diphtheria_Vaccination_Coverage],
            'HIV_AIDS_Prevalence_Rate': [HIV_AIDS_Prevalence_Rate],
            'Gross_Domestic_Product': [Gross_Domestic_Product],
            'Total_Population': [Total_Population],
            'Thinness': [Thinness],
            'Nation': [Nation],
            'Country_Category': [Country_Category]
        })

        # Get prediction
        prediction_result = prediction(new_data)

        # Display result
        st.success(f"Predicted Life Expectancy: {prediction_result[0]:.2f} years")

if __name__ == "__main__":
    main()
