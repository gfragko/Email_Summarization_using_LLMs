import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import fitz  # PyMuPDF for PDF text extraction

# Initialize the SentenceTransformer model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define a function to generate embeddings
def embed_text(texts):
    return embedding_model.encode(texts, convert_to_numpy=True)

# Initialize Chroma DB
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_pdf_pages"  # Directory to store the database files
))

# Create or load a Chroma collection
collection_name = "pdf_pages"
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=embed_text  # Custom embedding function
)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(pdf_document)):
        page_text = pdf_document.load_page(page_num).get_text("text")
        page_id = f"{os.path.basename(pdf_path)}_page_{page_num+1}"
        pages.append({"id": page_id, "text": page_text})
    return pages

# Add pages of a PDF file to the collection
pdf_path = "example.pdf"  # Path to the PDF file
pdf_pages = extract_text_from_pdf(pdf_path)

for page in pdf_pages:
    collection.add(
        ids=[page["id"]],
        documents=[page["text"]],
        metadatas=[{"source": pdf_path, "page": page["id"]}]
    )

print("PDF pages added to the Chroma database!")

# Query the database
query_text = "What is the definition of democracy?"
query_embedding = embed_text([query_text])

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3,  # Number of top results to retrieve
)

# Display the results
for doc, metadata in zip(results["documents"], results["metadatas"]):
    print(f"Source: {metadata['source']}\nPage ID: {metadata['page']}\nContent:\n{doc}\n")
