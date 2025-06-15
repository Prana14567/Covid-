import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="COVID-19 Forecast", layout="wide")

st.title("🦠 COVID-19 Confirmed Cases Forecast")
st.markdown("Built using **Facebook Prophet** and **Streamlit**")


@st.cache_data
def load_data():
    df = pd.read_csv("covid_19_data.csv")
    df = df.rename(columns={
        'ObservationDate': 'Date',
        'Province/State': 'State',
        'Country/Region': 'Country',
        'Last Update': 'Last_Update'
    })
    df['Date'] = pd.to_datetime(df['Date'])
    df['Confirmed'] = df['Confirmed'].fillna(0)
    return df

df = load_data()


country_list = sorted(df['Country'].dropna().unique())
selected_country = st.sidebar.selectbox("🌍 Select Country", country_list, index=country_list.index("India"))
selected_year = st.sidebar.selectbox("📅 Select Forecast Year", [2020, 2021, 2022, 2023, 2024, 2025], index=2)

# Filter and group data
df_country = df[df['Country'] == selected_country]
df_country = df_country.groupby('Date')['Confirmed'].sum().reset_index()
df_prophet = df_country.rename(columns={'Date': 'ds', 'Confirmed': 'y'})


model = Prophet()
model.fit(df_prophet)


years_ahead = selected_year - df_prophet['ds'].dt.year.max()
periods = years_ahead * 365 if years_ahead > 0 else 0
future = model.make_future_dataframe(periods=periods, freq='D')

forecast = model.predict(future)


forecast['ds'] = pd.to_datetime(forecast['ds'])
forecast_year = forecast[forecast['ds'].dt.year == selected_year]


fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(forecast_year['ds'], forecast_year['yhat'], label='Predicted')
ax.fill_between(forecast_year['ds'], forecast_year['yhat_lower'], forecast_year['yhat_upper'], color='skyblue', alpha=0.4)
ax.set_title(f'COVID-19 Forecast for {selected_country} ({selected_year})')
ax.set_xlabel("Date")
ax.set_ylabel("Confirmed Cases")
ax.grid(True)
ax.legend()
st.pyplot(fig)


with st.expander("📊 Show Forecast Data"):
    st.dataframe(forecast_year[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].reset_index(drop=True))
