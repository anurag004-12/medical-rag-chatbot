import os

from flask import Flask, render_template, request
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from src.prompt import prompt

from src.intent import detect_intent

# ----------------------------------------------------
# Load Environment Variables
# ----------------------------------------------------

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# ----------------------------------------------------
# Flask App
# ----------------------------------------------------

app = Flask(__name__)

# ----------------------------------------------------
# Embedding Model
# ----------------------------------------------------

print("Loading Embedding Model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding Model Loaded.")

# ----------------------------------------------------
# Pinecone
# ----------------------------------------------------

print("Connecting to Pinecone...")

pc = Pinecone(api_key=PINECONE_API_KEY)

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
)

print("Connected to Pinecone.")

# ----------------------------------------------------
# Retriever
# ----------------------------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k":3}
)

# ----------------------------------------------------
# Gemini
# ----------------------------------------------------

print("Loading Gemini...")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

print("Gemini Ready.")

# ----------------------------------------------------
# Document Chain
# ----------------------------------------------------

document_chain = create_stuff_documents_chain(
    llm,
    prompt
)

# ----------------------------------------------------
# Retrieval Chain
# ----------------------------------------------------

retrieval_chain = create_retrieval_chain(
    retriever,
    document_chain
)

print("RAG Pipeline Ready.")

# ----------------------------------------------------
# Home Page
# ----------------------------------------------------

@app.route("/")
def index():
    return render_template("chat.html")

# ----------------------------------------------------
# Chat API
# ----------------------------------------------------


@app.route("/get", methods=["POST"])
def chat():

    question = request.form["msg"].strip()

    print(f"\nUser : {question}")

    # --------------------------
    # Intent Detection
    # --------------------------

    intent = detect_intent(question)

    if intent == "greeting":
        return (
            "👋 Hello! I'm your AI Medical Assistant.\n\n"
            "I can answer questions based on the medical documents.\n"
            "For example:\n"
            "• What is diabetes?\n"
            "• Explain asthma.\n"
            "• What causes hypertension?"
        )

    elif intent == "thanks":
        return "😊 You're welcome! Feel free to ask another medical question."

    elif intent == "farewell":
        return "👋 Goodbye! Stay healthy and have a wonderful day!"

    # --------------------------
    # Medical Question
    # --------------------------

    response = retrieval_chain.invoke(
        {
            "input": question
        }
    )

    answer = response["answer"]

    print(f"Bot : {answer}")

    return answer

# ----------------------------------------------------
# Run App
# ----------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )