import streamlit as st
import pandas as pd
from fbprophet import Prophet
import time

df = pd.read_csv('example_wp_log_peyton_manning.csv')
st.write(df)

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
st.write(future)
start = time.time()
forecast = m.predict(future)
end = time.time()
#forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
st.write(forecast)

st.write(f"Elapsed time: {end - start}")
