import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from src.helper import (
    load_pdf,
    text_split,
    download_hugging_face_embeddings,
)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")


# ============================================================
# Step 1: Load PDF
# ============================================================

print("Loading PDF...")

documents = load_pdf("data")

print(f"Loaded {len(documents)} pages")


# ============================================================
# Step 2: Split PDF into chunks
# ============================================================

print("Splitting text...")

text_chunks = text_split(documents)

print(f"Created {len(text_chunks)} chunks")


# ============================================================
# Step 3: Load embedding model
# ============================================================

print("Loading embedding model...")

embeddings = download_hugging_face_embeddings()

print("Embedding model loaded")


# ============================================================
# Step 4: Connect to Pinecone
# ============================================================

print("Connecting to Pinecone...")

pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = pc.list_indexes().names()

print("Available Indexes:", existing_indexes)

if INDEX_NAME not in existing_indexes:
    raise ValueError(
        f"Index '{INDEX_NAME}' does not exist."
    )

print("Connected to Pinecone")


# ============================================================
# Step 5: Generate unique IDs
# ============================================================

print("Generating unique IDs...")

ids = []

for i, doc in enumerate(text_chunks):

    page = doc.metadata.get("page", 0)

    doc_id = f"medical-{page}-{i}"

    doc.metadata["id"] = doc_id

    ids.append(doc_id)

print(f"Generated {len(ids)} unique IDs")


# ============================================================
# Step 6: Store embeddings in Pinecone
# ============================================================

print("Uploading vectors...")

vectorstore = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=INDEX_NAME,
    ids=ids,
)

print("===================================")
print("Vector Store Created Successfully!")
print("===================================")