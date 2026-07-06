from src.helper import *

docs = load_pdf("data")

print(len(docs))

chunks = text_split(docs)

print(len(chunks))

emb = download_hugging_face_embeddings()

print(len(emb.embed_query("Hello")))