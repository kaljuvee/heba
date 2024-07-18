import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Set up OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is being loaded correctly
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables")

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, openai_api_key=openai_api_key)

# Initialize the embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to get embeddings for a list of texts
def get_embeddings(texts):
    return embeddings.embed_documents(texts)

# Function to find the most similar question and its answer
def find_similar_question(new_question, df, question_embeddings):
    new_question_embedding = get_embeddings([new_question])[0]
    similarities = cosine_similarity([new_question_embedding], question_embeddings)[0]
    most_similar_index = np.argmax(similarities)
    most_similar_question = df.iloc[most_similar_index]
    return most_similar_question['Question'], most_similar_question['Answer']

# Function to process a question
def process_question(question, qa_chain, df, question_embeddings):
    answer = qa_chain.run(question)
    if "Ma ei tea" in answer or "Ma pole kindel" in answer:
        # Perform similarity search if exact answer is not found
        similar_question, similar_answer = find_similar_question(question, df, question_embeddings)
        st.write("Ei leidnud täpset vastust. Siin on sarnane küsimus ja selle vastus:")
        st.write(f"Kõige sarnasem küsimus: {similar_question}")
        st.write(f"Soovitatav vastus: {similar_answer}")
    else:
        st.write("Vastus:", answer)

# Streamlit app
st.title("Menstastic AI - Vaimse Heaolu Assistent (RAG)")

# Sample questions
sample_questions = [
    "Olen tundnud stressi, kas saate mind aidata?", 
    "Kuidas ma saan hallata häirivaid mõtteid?", 
    "Ma jään mineviku sündmustele mõtlema, kuidas ma saan peatada?",
    "Millised on head viisid lõõgastumiseks?", 
    "Mul on raske leida aega iseendale, mis on teie nõuanne?"
]

# Create columns for sample questions
num_columns = 3
num_questions = len(sample_questions)
num_rows = (num_questions + num_columns - 1) // num_columns
columns = st.columns(num_columns)

# File uploader
uploaded_file = st.file_uploader("Vali CSV fail", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("CSV fail edukalt üles laaditud!")

    # Create vector store and QA chain
    loader = DataFrameLoader(df, page_content_column="Question")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(texts, embeddings)

    prompt_template = PromptTemplate(
        template="Kasuta järgmist konteksti, et vastata vaimse heaolu küsimusele: {context}\nKüsimus: {question}\nVastus. Küsige alati kasutajalt küsimust, et vestlust jätkata, kuid alles lõpus pärast mõne soovituse esitamist:",
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template}
    )

    # Get embeddings for all questions
    question_embeddings = get_embeddings(df['Question'].tolist())

    # Add buttons for sample questions
    for i in range(num_questions):
        col_index = i % num_columns
        row_index = i // num_columns

        with columns[col_index]:
            if st.button(sample_questions[i]):
                process_question(sample_questions[i], qa_chain, df, question_embeddings)

    # Question input for general questions
    st.subheader("Esitage küsimus vaimse heaolu kohta")
    question = st.text_input("Sisestage oma küsimus:")
    if st.button("Saada vastus"):
        if question:
            process_question(question, qa_chain, df, question_embeddings)
        else:
            st.warning("Palun sisestage küsimus.")
else:
    st.info("Palun laadige üles CSV fail alustamiseks.")
