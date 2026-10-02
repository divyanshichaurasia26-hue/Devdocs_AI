from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def create_chunks(text, chunk_size=2000):
    """
    Divide document text into smaller chunks.
    """

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        chunks.append(chunk)

    return chunks


def create_embeddings(chunks):
    """
    Convert document chunks into embeddings.
    """

    embeddings = model.encode(chunks)

    return embeddings


def retrieve_relevant_chunks(
    question,
    chunks,
    top_k=3
):
    """
    Find the document chunks that are
    semantically most similar to the question.
    """

    chunk_embeddings = model.encode(
        chunks
    )

    question_embedding = model.encode(
        [question]
    )

    similarities = cosine_similarity(
        question_embedding,
        chunk_embeddings
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        results.append({
            "chunk_index": int(index),
            "similarity": float(
                similarities[index]
            ),
            "text": chunks[index]
        })

    return results