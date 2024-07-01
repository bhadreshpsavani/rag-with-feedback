

#__import__('pysqlite3')
#import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI
from langchain_aws import ChatBedrock
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
import os
import streamlit as st


@st.cache_resource(show_spinner=False)
def initialize_chain(system_prompt, _memory):
    
    # llm = ChatOpenAI(temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo", streaming=True)
    llm = ChatBedrock(
        model_id="mistral.mistral-7b-instruct-v0:2",
        model_kwargs={"temperature": 0.1},
        region_name="us-east-1"
    )

    # Load markdown documents
    documents = []
    folder_path = "./markdown_files"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".md"):
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents.append(Document(page_content=content, metadata={}))
    
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        document_chunks = text_splitter.split_documents(documents)

        embeddings = BedrockEmbeddings(region_name="us-east-1")
        vectorstore = Chroma.from_documents(document_chunks, embeddings)

        # Initialize the ConversationalRetrievalChain with the system_prompt and memory parameters
        # qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory)
        qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=_memory)

        
        return qa
    else:
        raise Exception("No markdown files found. Please add markdown files in the specified folder to proceed.")
