import streamlit as st
import pandas as pd

st.title('Data Download')

# Function to read a CSV file
@st.cache
def read_df(file_path):
    return pd.read_csv(file_path)

# Function to convert DataFrame to CSV (as a download)
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Load the dataframes
df_ee = read_df('data/reversed_qa_ee.csv')  # Estonian data
df_en = read_df('data/reversed_qa_en.csv')  # English data

# Display information about the files
st.write("You can download Estonian and English datasets below:")

# Download buttons
st.download_button(
    label="Download Estonian Questionnaire Data (CSV)",
    data=convert_df_to_csv(df_ee),
    file_name='reversed_qa_ee.csv',
    mime='text/csv',
)

st.download_button(
    label="Download English Questionnaire Data (CSV)",
    data=convert_df_to_csv(df_en),
    file_name='reversed_qa_en.csv',
    mime='text/csv',
)
