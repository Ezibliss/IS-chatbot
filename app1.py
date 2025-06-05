import os
os.environ["TORCH_LOAD_META_TENSORS"] = "0"
import streamlit as st
import asyncio
from langchain.memory import ConversationBufferMemory
import openai
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM


pdf_loader = PyPDFLoader("INTERNAL SECURITY.pdf")
documents = pdf_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
db = FAISS.from_documents(documents = chunks, embedding = embeddings)
print("FAISS vector store created!")

llm = Ollama(model="llama3.2")

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template('''Act like an Instructor given 
the following conversation and a follow-up question, rephrase the follow-up question 
to be a standalone question.

Chat History:
{chat_history}
Follow-up Input:
{question}
Standalone questions: ''')

# Memory to store the conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(),
    condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    memory=memory,
    return_source_documents=False,
    verbose=False
)


try:
    asyncio.get_running_loop()
except RuntimeError:  # No loop is running
    asyncio.set_event_loop(asyncio.new_event_loop())


st.set_page_config(page_title="Your Interactive Learning Hub", page_icon="📚")

st.title("🌟 Welcome to Your Personal Tutor Bot!")
st.subheader("Your AI-powered assistant for all your learning needs.")

query = st.text_input("What do you want to know about IS:")



if query:
    with st.spinner("One moment please..."):
        try:
            response = qa.run({
                "question": query,
                "chat_history": []
            })
            st.write(response)
        except Exception as e:
            st.error(f"I'm sorry the answer is not contained in this NDA IS Precis : {e}")




