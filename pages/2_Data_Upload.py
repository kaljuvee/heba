import streamlit as st
import pandas as pd
import base64
import io

st.title('Data Upload')

# Function to convert DataFrame to CSV (as a download)
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Function to convert DataFrame to Excel (as a download)
def convert_df_to_excel(df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False)
    writer.save()
    processed_data = output.getvalue()
    return processed_data

# Upload file and recognize file type
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# Check if a file has been uploaded
if uploaded_file:
    # If CSV, read with pandas
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        st.write('Dataframe (CSV):')
    # If Excel, read with pandas
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
        st.write('Dataframe (Excel):')
    
    # Display the dataframe
    st.dataframe(df)

    # Download buttons
    st.write('Download the data as CSV or Excel:')
    csv = convert_df_to_csv(df)  # Convert DataFrame to CSV
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name='data.csv',
        mime='text/csv',
    )

    excel = convert_df_to_excel(df)  # Convert DataFrame to Excel
    st.download_button(
        label="Download as Excel",
        data=excel,
        file_name='data.xlsx',
        mime='application/vnd.ms-excel',
    )
else:
    st.write("Please upload a file to view and download the data.")
