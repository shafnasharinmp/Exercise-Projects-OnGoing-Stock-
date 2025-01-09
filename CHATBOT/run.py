from pdf_extract import extract_text_from_pdfs
from embedding import generate_embeddings
from retrieval import retrieve_documents
from chatbot import generate_response

pdf_paths = ['file1.pdf', 'file2.pdf']
documents = extract_text_from_pdfs(pdf_paths)


document_vectors = generate_embeddings(documents)


query = "What is machine learning?"


retrieved_docs = retrieve_documents(query, document_vectors, documents, top_k=2)
context = " ".join(retrieved_docs)

response = generate_response(context, query)
print("Chatbot Response:", response)
