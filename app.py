import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel 

# ... Gerekli diğer importlar ...
from pydantic import BaseModel

class validation(BaseModel):
    prompt: str
# Yardımcı Fonksiyonlar (Birinci Kod Bloğundan)
def get_pdf_text(pdf_docs):
  text=""
  for pdf in pdf_docs:
    pdf_reader= PdfReader(pdf)
    for page in pdf_reader.pages:
      text+= page.extract_text()
  return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):  
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# FastAPI Fonksiyonları (İkinci Kod Bloğundan)
def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def load_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)
    return llm

def get_conversational_chain():
    prompt = set_custom_prompt()
    llm = load_llm()
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

# ... Validation sınıfı (Pydantic - İkinci Koddan) ...
app = FastAPI()
# FastAPI uç noktası (İkinci Kod Bloğundan)
@app.post("/llm_on_cpu")
async def final_result(item: validation):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    db = FAISS.load_local("vectorstore/db_faiss", embeddings)
    docs = db.similarity_search(item.prompt)
    chain = get_conversational_chain()
    response = chain({"input_documents":docs, "question": item.prompt}, return_only_outputs=False)
    response['output_text'] = response['output_text'].replace("\n", " ")
    return response

# Streamlit Arayüzü
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini!")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        # Kullanıcı sorgusuyla FastAPI Uç Noktasına AJAX Çağrısı
        response = requests.post("http://localhost:8000/llm_on_cpu", json={"prompt": user_question})
        response = response.json()

        st.write("Reply: ", response["output_text"])

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
