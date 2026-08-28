import streamlit as st
import requests

st.title("Retail Out-of-Stock Prediction Dashboard")

st.write("Enter today's store-product details to predict tomorrow's out-of-stock risk.")

units_sold = st.number_input("Units Sold", min_value=0.0, value=80.0)
units_ordered = st.number_input("Units Ordered", min_value=0.0, value=100.0)
price = st.number_input("Price", min_value=0.0, value=50.0)
discount = st.number_input("Discount", min_value=0.0, value=10.0)
promotion = st.selectbox("Promotion Running?", [0, 1])
competitor_pricing = st.number_input("Competitor Pricing", min_value=0.0, value=55.0)
epidemic = st.selectbox("Epidemic Flag", [0, 1])
demand = st.number_input("Demand", min_value=0.0, value=90.0)
prev_inventory_level = st.number_input("Previous Day Inventory Level", min_value=0.0, value=20.0)
prev_units_sold = st.number_input("Previous Day Units Sold", min_value=0.0, value=75.0)
rolling_7d_units_sold = st.number_input("7-Day Rolling Avg Units Sold", min_value=0.0, value=85.0)
day_of_week = st.selectbox("Day of Week (0=Mon, 6=Sun)", [0, 1, 2, 3, 4, 5, 6])
month = st.selectbox("Month", list(range(1, 13)))
category = st.selectbox("Category", ["Groceries", "Electronics", "Clothing", "Furniture", "Toys"])
region = st.selectbox("Region", ["North", "South", "East", "West"])
weather_condition = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Cloudy", "Snowy"])
seasonality = st.selectbox("Seasonality", ["Summer", "Winter", "Spring", "Autumn"])

if st.button("Predict"):
    input_data = {
        "Units_Sold": units_sold,
        "Units_Ordered": units_ordered,
        "Price": price,
        "Discount": discount,
        "Promotion": promotion,
        "Competitor_Pricing": competitor_pricing,
        "Epidemic": epidemic,
        "Demand": demand,
        "Prev_Inventory_Level": prev_inventory_level,
        "Prev_Units_Sold": prev_units_sold,
        "Rolling_7d_Units_Sold": rolling_7d_units_sold,
        "Day_Of_Week": day_of_week,
        "Month": month,
        "Category": category,
        "Region": region,
        "Weather_Condition": weather_condition,
        "Seasonality": seasonality
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
    result = response.json()

    st.write("Out-of-stock probability:", result["out_of_stock_probability"])

    if result["prediction"] == 1:
        st.error("Prediction: Likely to be OUT OF STOCK tomorrow")
    else:
        st.success("Prediction: Likely to remain IN STOCK tomorrow")