from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Initialize the SentenceTransformer model (same model used during database creation)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define a function to generate embeddings (same as during database creation)
def embed_text(texts):
    return embedding_model.encode(texts, convert_to_numpy=True)

# Initialize Chroma client and load the existing database
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_pdf_pages"  # Directory where the database is stored
))

# Load the existing collection
collection_name = "pdf_pages"
collection = client.get_collection(name=collection_name)

# Query text
query_text = "What is the definition of democracy?"
query_embedding = embed_text([query_text])

# Query the database
results = collection.query(
    query_embeddings=query_embedding,
    n_results=3  # Number of top results to retrieve
)

# Display the results
for doc, metadata in zip(results["documents"], results["metadatas"]):
    print(f"Source: {metadata['source']}\nPage ID: {metadata['page']}\nContent:\n{doc}\n")
