import streamlit as st
import pickle 
import sklearn
# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
import pickle
import pickle
import sklearn


st.title("Retail Demand Prediction App")
st.markdown("Predict product demand based on various factors")

# Input fields
Store_ID = st.radio("1. Select the Store", ["S001", "S002", "S003", "S004", "S005"])
Product_ID = st.radio("2. Select the Product", ["P001", "P002", "P003", "P004", "P005", "P006", "P007", "P008", "P009", "P010", 
                                              "P011", "P012", "P013", "P014", "P015", "P016", "P017", "P018", "P019", "P020"])
Category = st.radio("3. Select the Category", ["Furniture", "Toys", "Clothing", "Groceries", "Electronics"])
Region = st.radio("4. Select the Region", ["North", "South", "West", "East"])
Inventory_Level = st.number_input("5. Select the Inventory Level", min_value=50, max_value=500)
Units_Sold = st.number_input("6. Select the Units Sold", min_value=0, max_value=499)
Units_Ordered = st.number_input("7. Select the Units Ordered", min_value=20, max_value=200)
Demand_Forecast = st.number_input("8. Select the Demand Forecast", min_value=-9.99, max_value=518.55)
Discount = st.number_input("9. Select the Discount", min_value=0, max_value=20)
Weather_Condition = st.radio("10. Select the Weather Condition", ["Sunny", "Rainy", "Snowy", "Cloudy"])
Holiday_or_Promotion = st.radio("11. Select the Holiday/Promotion", ["0", "1"])
Competitor_Pricing = st.number_input("12. Select the Competitor Pricing", min_value=5.03, max_value=104.94)
Seasonality = st.radio("13. Select the Seasonality", ["Spring", "Summer", "Winter", "Autumn"])

# Load the trained pipeline model
with open("Retail_Regree.pkl","rb") as f:
    final = pickle.load(f)

if st.button("Submit the details"):
    # Create a DataFrame with the input data in the correct order
    input_data = pd.DataFrame([[Store_ID, Product_ID, Category, Region, Inventory_Level,
                             Units_Sold, Units_Ordered, Demand_Forecast, Discount,
                             Weather_Condition, Holiday_or_Promotion, Competitor_Pricing,
                             Seasonality]],
                           columns=['Store ID', 'Product ID', 'Category', 'Region', 'Inventory Level',
                                    'Units Sold', 'Units Ordered', 'Demand Forecast', 'Discount',
                                    'Weather Condition', 'Holiday/Promotion', 'Competitor Pricing',
                                    'Seasonality'])
    
    # Make prediction
    pred = final.predict(input_data)
    st.success(f"Predicted Demand: {pred[0]:.2f}")
