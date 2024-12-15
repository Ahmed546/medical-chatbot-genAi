from src.helper import load_pdf_file,text_split,download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain import vectorstores
from dotenv import load_dotenv

import os

load_dotenv()

PINECONE_API_KEY= os.environ.get('PINECONE_API_KEY')

extracted_data = load_pdf_file(data='data/')

text_chunks = text_split(extracted_data)

embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define the index name and configuration
index_name = "medical-chatbot"

try:
    # Check if the index already exists
    existing_indexes = pc.list_indexes()
    if index_name in existing_indexes:
        print(f"Index '{index_name}' already exists. Proceeding to the next step...")
    else:
        # Create the index if it doesn't exist
        pc.create_index(
            name=index_name,
            dimension=384,  # Replace with your model dimensions
            metric="cosine",  # Replace with your model metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Index '{index_name}' created successfully.")
except Exception as e:
    print(f"An error occurred: {e}")


dosearch= vectorstores.Pinecone.from_documents(
    documents = text_chunks,
    index_name = index_name,
    embedding= embeddings,
)
