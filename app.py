from flask import Flask,render_template,jsonify,request
from src.helper import download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain import vectorstores
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
HF_Token= os.environ.get('HF_Token')

embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define the index name and configuration
index_name = "medical-chatbot"

docsearch = vectorstores.Pinecone.from_existing_index(
    index_name = index_name,
    embedding=embeddings,
)

PROMPT = PromptTemplate(template=prompt_template, input_veriables=['context','question'])
chain_type_kwargs={"prompt":PROMPT}

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

# Define model_kwargs
model_kwargs = {
    "max_length": 128
}

# Initialize the model with the updated parameters
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    token=HF_Token,
    temperature=0.7,  # Pass temperature explicitly
    model_kwargs=model_kwargs
)

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)


@app.route('/')
def index():
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080 ,debug=True)