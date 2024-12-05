import argparse
import os
import shutil
from langchain_community.document_loaders import  PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from pdf2image import convert_from_path
import pytesseract
import PyPDF2
from langchain_huggingface import HuggingFaceEmbeddings

PDFS_PATH = "Dictionary_PDFs"
CHROMA_PATH = "Dictionary_DB"


# Initialize the SentenceTransformer model
embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Define a function to generate embeddings
# def embed_text(texts):
#     return embedding_model.encode(texts, convert_to_numpy=True)

def load_documents():
    """Load documents, using Tesseract OCR for scanned PDFs."""
    documents = []
    for filename in os.listdir(PDFS_PATH):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDFS_PATH, filename)
            print(f"Processing PDF: {pdf_path}")
            if not is_pdf_scanned(pdf_path):                
                # Try loading the document using PyPDFDirectoryLoader
                # document_loader = PyPDFDirectoryLoader(pdf_path)
                document_loader = PyPDFLoader(pdf_path)
                extracted_documents = document_loader.load()
                # print(extracted_documents)
                documents.extend(extracted_documents)  # Add text documents to the list
            else:
                # print("helloo")
                # If no text is found, use OCR via Tesseract
                extracted_text = extract_text_from_image(pdf_path)
                if extracted_text.strip():
                    document = Document(page_content=extracted_text, metadata={"source": filename})
                    documents.append(document)

    return documents



def is_pdf_scanned(pdf_path):
    """Function to read and extract text from PDF"""
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text() + "\n"
    
    print("trying to find text...")   
    if not text.strip():
        print("Pdf is a scanned document...")
        return True
    else:
        print("PDF with text.")     
    
    return False


def extract_text_from_image(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for page_number, img in enumerate(images):
        print(f"Processing page {page_number + 1}/{len(images)}...")
        # Use Tesseract to extract text from the image
        page_text = pytesseract.image_to_string(img)
        # Append the text from this page to the overall text
        text += page_text + "\n"
        # print(text)
    
    return text



def split_documents(documents: list[Document]):
    '''Splitting pdfs into chunks to create a better DB'''    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


#In order to not create the db from scratch every time the script runs
#we use this function to id each pdf and pdf page so that we are able 
# to add to the db instead of re-initializing it
def calculate_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    last_page_id = None
    current_chunk_index = 0
    
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        curr_page_id = f"{source}:{page}"
        
        # If the page ID is the same as the last one, increment the index.
        if curr_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
            
        # Calculate the chunk ID.
        chunk_id = f"{curr_page_id}:{current_chunk_index}"
        last_page_id = curr_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id
           
    return chunks




def add_to_chroma(chunks: list[Document]):
    # Load or create the database
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embedding_function  
    )  
    
    # Calculate Page IDs for all chunks
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    # get the existing documents
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    
    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        
    else:
        print("✅ No new documents to add")
    

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    
def main():
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    print("============================================")
    print("finding incoterms relevance...")
    dictionary_db = Chroma(persist_directory="Dictionary_DB", embedding_function=embedding_function)
    # dictionary_results = dictionary_db.similarity_search("cto", k=5)
    #for idx, result in enumerate(results):
    # print(f"Result {idx+1}: {result.page_content}")
    results_with_scores = dictionary_db.similarity_search_with_score("MB ", k=5)
    for doc, score in results_with_scores:
        print(f"Document: {doc.page_content}, Similarity Score: {score}")
        print("=========================================================")
    
    
if __name__ == "__main__":
    main()