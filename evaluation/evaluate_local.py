import ast
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Load Predictions
# -------------------------

df = pd.read_csv("evaluation/predictions.csv")

df["retrieved_contexts"] = df["retrieved_contexts"].apply(ast.literal_eval)

# -------------------------
# Load Embedding Model
# -------------------------

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

similarities = []
coverages = []
answer_lengths = []

# -------------------------
# Evaluate
# -------------------------

for _, row in df.iterrows():

    reference = row["reference"]
    response = row["response"]

    context = " ".join(row["retrieved_contexts"])

    # Semantic Similarity

    ref_embedding = model.encode(reference)
    res_embedding = model.encode(response)

    similarity = cosine_similarity(
        [ref_embedding],
        [res_embedding]
    )[0][0]

    similarities.append(round(float(similarity), 3))

    # Lexical Context Coverage

    answer_words = set(response.lower().split())
    context_words = set(context.lower().split())

    coverage = len(answer_words & context_words) / max(len(answer_words), 1)

    coverages.append(round(float(coverage), 3))

    # Answer Length

    answer_lengths.append(len(response.split()))

# -------------------------
# Save Report
# -------------------------

df["semantic_similarity"] = similarities
df["lexical_context_coverage"] = coverages
df["answer_length"] = answer_lengths

df.to_csv(
    "evaluation/evaluation_report.csv",
    index=False
)

# -------------------------
# Summary
# -------------------------

avg_similarity = sum(similarities) / len(similarities)
avg_coverage = sum(coverages) / len(coverages)
avg_length = sum(answer_lengths) / len(answer_lengths)

with open(
    "evaluation/evaluation_summary.txt",
    "w"
) as f:

    f.write("Medical RAG Chatbot Evaluation\n\n")

    f.write(f"Questions Evaluated : {len(df)}\n")
    f.write(f"Average Semantic Similarity : {avg_similarity:.3f}\n")
    f.write(f"Average Lexical Context Coverage : {avg_coverage:.3f}\n")
    f.write(f"Average Answer Length : {avg_length:.1f} words\n")

print("\nEvaluation Complete!")

print(f"Average Semantic Similarity : {avg_similarity:.3f}")
print(f"Average Lexical Context Coverage : {avg_coverage:.3f}")
print(f"Average Answer Length : {avg_length:.1f} words")

# -------------------------
# Similarity Graph
# -------------------------

plt.figure(figsize=(8,4))
plt.bar(range(1, len(similarities)+1), similarities)
plt.xlabel("Question")
plt.ylabel("Semantic Similarity")
plt.title("Semantic Similarity per Question")
plt.tight_layout()
plt.savefig("evaluation/similarity.png")
plt.close()

# -------------------------
# Coverage Graph
# -------------------------

plt.figure(figsize=(8,4))
plt.bar(range(1, len(coverages)+1), coverages)
plt.xlabel("Question")
plt.ylabel("Lexical Context Coverage")
plt.title("Lexical Context Coverage per Question")
plt.tight_layout()
plt.savefig("evaluation/coverage.png")
plt.close()

# -------------------------
# Answer Length Graph
# -------------------------

plt.figure(figsize=(8,4))
plt.bar(range(1, len(answer_lengths)+1), answer_lengths)
plt.xlabel("Question")
plt.ylabel("Words")
plt.title("Answer Length")
plt.tight_layout()
plt.savefig("evaluation/answer_length.png")
plt.close()

print("\nGraphs saved in evaluation/")