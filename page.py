import pickle 
import numpy as np 
import pandas as pd 
import streamlit as st
import plotly.express as px

from custom_transformers import handle_outliers
from custom_transformers import label_encode_and_scale
from custom_transformers import fillna_median
from custom_transformers import square

# Load model
loaded_model = pickle.load(open('life_expectancy.sav', 'rb'))
model = loaded_model['pipeline']

# Streamlit App Configuration
st.set_page_config(page_title="Life Expectancy App", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Life Expectancy Prediction", "Data Analysis"])

# 🏠 Home Page
if page == "Home":
    st.title("Welcome to the Life Expectancy App 🌍")
    st.markdown("""
    - **Life Expectancy Prediction**: Enter country details to predict life expectancy.
    - **Data Analysis**: Explore and visualize global life expectancy trends.
    """)
    st.image("https://media.istockphoto.com/id/2062608145/photo/a-female-nurse-caregiver-holds-hands-to-encourage-and-comfort-an-elderly-woman-for-care-and.jpg?s=1024x1024&w=is&k=20&c=3pzOtkanA5PKX4n9zk29jXmH0wwrZRSkZ4ha1HhEU7Y=", use_container_width=True)

# 📊 Data Analysis Page
elif page == "Data Analysis":
    st.title("📊 Life Expectancy Data Analysis")

    # Load the dataset
    df = pd.read_csv("IEEE.csv")  # Update with your actual file

    # Strip spaces from column names
    df.columns = df.columns.str.strip()

    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Ensure necessary columns exist
    if "Survey_Year" not in df.columns or "Country_Category" not in df.columns or "Life_Expectancy_Years" not in df.columns:
        st.error("Required columns not found! Check column names.")
    else:
        # Convert 'Survey_Year' to numeric
        df["Survey_Year"] = pd.to_numeric(df["Survey_Year"], errors="coerce").astype("Int64")

        st.sidebar.header("Filters")
        analysis_option = st.sidebar.selectbox(
            "Choose Analysis Option:",
            ["Overall Data Over All Years", "Specific Country Over All Years", "Specific Country Within a Specific Year"]
        )

        # Option 1: Overall Data Over All Years
        if analysis_option == "Overall Data Over All Years":
            st.subheader("Overall Data Over All Years")
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col not in ["Survey_Year", "Life_Expectancy_Years"]]

            if numeric_cols:
                selected_feature = st.selectbox("Select a feature to visualize:", numeric_cols)

                category_data = df.groupby("Country_Category")[selected_feature].mean().reset_index()
                fig = px.pie(category_data, values=selected_feature, names="Country_Category",
                             title=f"Distribution of {selected_feature} by Country Category",
                             color_discrete_sequence=px.colors.qualitative.Pastel)

                yearly_data = df.groupby("Survey_Year", as_index=False)[numeric_cols].mean()
                fig2 = px.line(yearly_data, x="Survey_Year", y=selected_feature,
                               title=f"{selected_feature} Trend Over All Years")

                top_10_countries = df.groupby("Nation")[selected_feature].mean().nlargest(10).reset_index()
                fig3 = px.bar(top_10_countries, x="Nation", y=selected_feature, text=selected_feature,
                              title=f"Top 10 Countries by {selected_feature}",
                              color="Nation",
                              color_discrete_sequence=px.colors.qualitative.Vivid)

                fig4 = px.scatter(df, x=selected_feature, y="Life_Expectancy_Years",
                                  title=f"Correlation Between {selected_feature} and Life Expectancy",
                                  color="Nation", 
                                  size_max=10)

                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.plotly_chart(fig2, use_container_width=True)

                st.plotly_chart(fig3, use_container_width=True)
                st.plotly_chart(fig4, use_container_width=True)

            else:
                st.warning("No numeric columns found for visualization.")

        elif analysis_option == "Specific Country Over All Years":
            selected_country = st.sidebar.selectbox("Select a country:", df["Nation"].unique())
            st.subheader(f"Data for {selected_country} Over All Years")
            filtered_data = df[df["Nation"] == selected_country]
            numeric_cols = filtered_data.select_dtypes(include=['number']).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != "Survey_Year"]

            if numeric_cols:
                selected_feature = st.selectbox("Select a feature to plot:", numeric_cols)
                filtered_data = filtered_data.groupby("Survey_Year", as_index=False)[selected_feature].mean()
                filtered_data = filtered_data.sort_values(by="Survey_Year")
                fig = px.line(filtered_data, x="Survey_Year", y=selected_feature,
                              title=f"{selected_feature} for {selected_country} Over All Years",
                              markers=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns found for visualization.")

        elif analysis_option == "Specific Country Within a Specific Year":
            selected_country = st.sidebar.selectbox("Select a country:", df["Nation"].unique())
            selected_year = st.sidebar.selectbox("Select a year:", df["Survey_Year"].dropna().unique())
            st.subheader(f"Data for {selected_country} in {selected_year}")
            filtered_data = df[(df["Nation"] == selected_country) & (df["Survey_Year"] == selected_year)]

            if not filtered_data.empty:
                st.subheader("Detailed Data")
                transposed_data = filtered_data.T.reset_index()
                transposed_data.columns = ["Feature", "Value"]
                st.dataframe(transposed_data, use_container_width=True)
            else:
                st.warning("No data found for the selected filters.")

# 🔮 Life Expectancy Prediction Page
elif page == "Life Expectancy Prediction":
    st.title("🌍 Life Expectancy Prediction")

    # Input fields
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

    if st.button("Predict Life Expectancy"):
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

        prediction_result = model.predict(new_data)
        st.success(f"Predicted Life Expectancy: {prediction_result[0]:.2f} years")
