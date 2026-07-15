import os
import time
import pandas as pd
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from src.prompt import prompt

# ---------------------------
# Load Environment Variables
# ---------------------------

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# ---------------------------
# Embedding Model
# ---------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------
# Pinecone Vector Store
# ---------------------------

pc = Pinecone(api_key=PINECONE_API_KEY)

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# ---------------------------
# Gemini LLM
# ---------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

# ---------------------------
# Build RAG Chain
# ---------------------------

document_chain = create_stuff_documents_chain(
    llm,
    prompt
)

retrieval_chain = create_retrieval_chain(
    retriever,
    document_chain
)

# ---------------------------
# Load Benchmark Dataset
# ---------------------------

df = pd.read_csv("evaluation/benchmark.csv")

# ---------------------------
# Store Results
# ---------------------------

questions = []
references = []
responses = []
retrieved_contexts = []
response_times = []

# ---------------------------
# Generate Answers
# ---------------------------

for i, row in df.iterrows():

    question = row["question"]

    print(f"\n{i+1}. {question}")

    while True:
        try:

            start = time.time()

            result = retrieval_chain.invoke(
                {"input": question}
            )

            end = time.time()

            break

        except Exception as e:

            print(e)
            print("Waiting 30 seconds...")
            time.sleep(30)

    answer = result["answer"]

    docs = result["context"]

    contexts = [
        doc.page_content
        for doc in docs
    ]

    questions.append(question)
    references.append(row["ground_truth"])
    responses.append(answer)
    retrieved_contexts.append(contexts)
    response_times.append(round(end - start, 3))

    print(f"Response Time: {round(end-start,3)} sec")

    # Avoid hitting Gemini RPM limit
    time.sleep(12)

# ---------------------------
# Save Predictions
# ---------------------------

prediction_df = pd.DataFrame({
    "user_input": questions,
    "reference": references,
    "response": responses,
    "retrieved_contexts": retrieved_contexts,
    "response_time": response_times
})

prediction_df.to_csv(
    "evaluation/predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")
print("File: evaluation/predictions.csv")