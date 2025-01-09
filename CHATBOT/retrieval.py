import faiss
import numpy as np

def build_faiss_index(document_vectors):
    dimension = document_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(document_vectors))
    return index

def retrieve_documents(query, document_vectors, documents, top_k=2):
    query_vector = embedder.encode([query], convert_to_tensor=False)
    index = build_faiss_index(document_vectors)
    distances, indices = index.search(np.array(query_vector), top_k)
    retrieved_docs = [documents[idx] for idx in indices[0]]
    return retrieved_docs
