from langchain.agents import AgentType
from langchain_community.agents import create_pandas_dataframe_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_models import ChatOpenAI
import streamlit as st
import pandas as pd
import os

# Initialize OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define model
model = 'gpt-4-1106-preview'

# Page configuration
st.set_page_config(page_title="Mentastic")
st.title("Mentastic AI")

# Function to read the DataFrame
@st.cache_data
def read_df(file_path):
    return pd.read_csv(file_path)

# Load the data
df = read_df('data/reversed_qa_en.csv')

# Function to process a question
def process_question(question):
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    llm = ChatOpenAI(
        temperature=0, model=model, openai_api_key=openai_api_key, streaming=True
    )

    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
        allow_dangerous_code=True  # Allow dangerous code execution
    )

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)

# Initialize or clear conversation history
if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display conversation history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Sample questions
sample_questions = ["I've been feeling stressed, can you help me?", "How can I manage disturbing thoughts?", "I keep ruminating on past events, how can I stop?",
                    "What are some good ways to relax?", "Finding time for myself is hard, what's your advice?"]

# Create columns for sample questions
num_columns = 3
num_questions = len(sample_questions)
num_rows = (num_questions + num_columns - 1) // num_columns
columns = st.columns(num_columns)

# Add buttons for sample questions
for i in range(num_questions):
    col_index = i % num_columns
    row_index = i // num_columns

    with columns[col_index]:
        if columns[col_index].button(sample_questions[i]):
            process_question(sample_questions[i])

# User input for new questions
container = st.container()
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Ask a question:", key='input')
        submit_button = st.form_submit_button(
