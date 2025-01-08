from sentence_transformers import SentenceTransformer

def generate_embeddings(documents):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    document_vectors = embedder.encode(documents, convert_to_tensor=False)
    return document_vectors
