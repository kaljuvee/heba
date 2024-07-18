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
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=openai_api_key)

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

# Streamlit app
st.title("Mental Wellbeing Assistant")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("CSV file uploaded successfully!")

    # Create vector store and QA chain
    loader = DataFrameLoader(df, page_content_column="Question")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(texts, embeddings)

    prompt_template = PromptTemplate(
        template="Use the following context to answer the question about mental wellbeing: {context}\nQuestion: {question}\nAnswer:",
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

    # Question input for general questions
    st.subheader("Ask a question about mental wellbeing")
    question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if question:
            answer = qa_chain.run(question)
            if "I don't know" in answer or "I'm not sure" in answer:
                # Perform similarity search if exact answer is not found
                similar_question, similar_answer = find_similar_question(question, df, question_embeddings)
                st.write("Couldn't find an exact answer. Here's a similar question and its answer:")
                st.write(f"Most Similar Question: {similar_question}")
                st.write(f"Suggested Answer: {similar_answer}")
            else:
                st.write("Answer:", answer)
        else:
            st.warning("Please enter a question.")
else:
    st.info("Please upload a CSV file to begin.")