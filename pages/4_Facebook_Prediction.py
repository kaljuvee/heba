import streamlit as st
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from prophet import Prophet

faker = Faker()

# Function to generate synthetic data
def generate_data(start_date, end_date):
    data = []
    for _ in range((end_date - start_date).days + 1):
        date = start_date + timedelta(days=_)
        data.append({
            "date": date,
            "user_id": faker.unique.random_int(min=1, max=1000),
            "name": faker.name(),
            "email": faker.email(),
            "gender": faker.random_element(elements=('Male', 'Female')),
            "birthday": faker.date_of_birth(tzinfo=None, minimum_age=18, maximum_age=70),
            "location": faker.address(),
            "page_id": faker.unique.random_int(min=1, max=100),
            "page_name": faker.company(),
            "about": faker.text(max_nb_chars=200),
            "category": faker.random_element(elements=('Entertainment', 'Business', 'Community')),
            "fan_count": faker.random_int(min=100, max=10000),
            "posts": faker.random_int(min=0, max=5),  # Number of posts as the target variable
        })
    return pd.DataFrame(data)

st.title("Facebook Data Forecasting")

st.sidebar.header("Data Generation Settings")
start_date = st.sidebar.date_input("Start date", datetime(2023, 12, 1))
end_date = st.sidebar.date_input("End date", datetime(2023, 12, 31))

if st.sidebar.button("Generate Data"):
    st.session_state['df'] = generate_data(start_date, end_date)

    # Encode categorical variables
    st.session_state['df']['gender_encoded'] = LabelEncoder().fit_transform(st.session_state['df']['gender'])
    st.session_state['df']['category_encoded'] = LabelEncoder().fit_transform(st.session_state['df']['category'])

if 'df' in st.session_state:
    st.write("Generated Data:", st.session_state['df'].head())

    # Select regressors
    available_regressors = ['gender_encoded', 'category_encoded', 'fan_count']
    selected_regressors = st.multiselect("Select Regressors:", available_regressors, default=available_regressors)

    if st.button("Rerun with Selected Regressors"):
        prophet_df = st.session_state['df'][['date', 'posts'] + selected_regressors].rename(columns={'date': 'ds', 'posts': 'y'})

        model = Prophet(daily_seasonality=True)
        for regressor in selected_regressors:
            model.add_regressor(regressor)

        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=30)
        future = future.merge(prophet_df.drop('y', axis=1), on='ds', how='left').fillna(method='ffill')

        forecast = model.predict(future)

        st.subheader("Forecast with Selected Regressors")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        st.subheader("Forecast Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)
