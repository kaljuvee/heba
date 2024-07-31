import streamlit as st
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Set up OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

# Function to save data to CSV
def save_to_csv(df, file_name):
    if not os.path.isfile(file_name):
        df.to_csv(file_name, index=False)
    else:
        df.to_csv(file_name, mode='a', header=False, index=False)

# Function to extract number from text using OpenAI
def extract_number(response_text):
    prompt = f"Extract the number from the following response (if the response is a word, convert it to a number between 0 and 5): \"{response_text}\""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts numbers from text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10
    )
    try:
        extracted_number = int(response.choices[0].message.content.strip())
        if 0 <= extracted_number <= 5:
            return extracted_number
    except ValueError:
        pass
    return None

# Initialize session state
if 'question_index' not in st.session_state:
    st.session_state.question_index = 0
if 'responses' not in st.session_state:
    st.session_state.responses = []

# WHO5 questions in Estonian
questions = [
    "1. Olen tundnud end rõõmsana ja heas tujus.",
    "2. Olen tundnud end rahulikuna ja lõdvestununa.",
    "3. Olen tundnud end energilise ja aktiivsena.",
    "4. Olen ärganud puhanuna ja värskena.",
    "5. Mu igapäevaelu on olnud täis asju, mis pakuvad huvi."
]

# Display the title
st.title("WHO-5 küsimustik")
st.write("Palun hinda viimase kahe nädala jooksul oma tundeid järgmiste väidete põhjal:")

# Function to handle form submission
def handle_submit():
    response = st.session_state.current_response
    extracted_number = extract_number(response)
    if extracted_number is not None:
        st.session_state.responses.append(extracted_number)
        st.session_state.question_index += 1
        st.success("Vastus on salvestatud!")
    else:
        st.error("Palun sisesta korrektne number vahemikus 0-5 või number sõnana.")

# Display questions and handle responses
if st.session_state.question_index < len(questions):
    current_question = questions[st.session_state.question_index]
    st.write(f"Küsimus {st.session_state.question_index + 1}/{len(questions)}:")
    st.write(current_question)
    
    # Use a form to handle input and submission
    with st.form(key=f'question_form_{st.session_state.question_index}'):
        st.text_input("Sinu vastus (0-5):", key='current_response')
        submit_button = st.form_submit_button(label='Saada')
        
        if submit_button:
            handle_submit()

else:
    st.write("Aitäh, et vastasid kõigile küsimustele!")
    # Convert responses to DataFrame and save to CSV
    df = pd.DataFrame([st.session_state.responses], columns=[q.split('. ', 1)[1] for q in questions])
    save_to_csv(df, 'who5_responses.csv')
    st.session_state.responses = []  # Reset responses
    st.session_state.question_index = 0  # Reset question index
    
    if st.button("Alusta uuesti"):
        st.experimental_rerun()

# Show the DataFrame
if st.checkbox("Näita vastuseid"):
    if os.path.isfile('who5_responses.csv'):
        df = pd.read_csv('who5_responses.csv')
        st.dataframe(df)
    else:
        st.write("Vastused puuduvad.")