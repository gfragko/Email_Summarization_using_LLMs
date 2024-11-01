import time
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain.schema.document import Document
from datetime import datetime
import emailHandler


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


REVIEW_TEMPLATE = """
I will provide a piece of email conversation along with the attached files and then a summary of these pieces of text.

---
email_text:
{email_text}

---
Attached files:
{attachments_text}

---
Summary:
{summary_text}

---
I want the summary text to contains all important details about the ids of the shipments and their statuses (if any) are present or other document ids or even changes of plans!
I want you to review how good the summary is based on the rules above. 
Provide comments on how it could be better and create a new summary.
Finally provide a new prompt to feed into Ollama that you think would achieve a better summarization (not just for this specific example). 


"""

EMB_FUNC = OllamaEmbeddings(model="llama3.1:latest")

# Initialize Chroma DBs for dictionary and email history
def init_DBs():
    dictionary_db = Chroma(persist_directory="Dictionary_DB", embedding_function=EMB_FUNC)
    email_history_db = Chroma(persist_directory="Email_History_DB", embedding_function=EMB_FUNC)
    # attachments_db = 
    return dictionary_db, email_history_db



# Function to retrieve context from both Chroma DBs
def retrieve_context(dictionary_db, email_history_db, query):
    # Retrieve the meaning of special terms from the dictionary
    print("find dictionary relevance")
    dictionary_results = dictionary_db.similarity_search(query, k=5)
    print("found")
    # Retrieve previous conversation history from email history
    print("find email_history_results relevance")
    email_history_results = email_history_db.similarity_search(query, k=5)

    return dictionary_results, email_history_results

# Function to retrieve context from both Chroma DBs
def retrieve_context_dictionary_db(dictionary_db, query):
    # Retrieve the meaning of special terms from the dictionary
    print("find dictionary relevance")
    dictionary_results = dictionary_db.similarity_search(query, k=5)
    print("found")
    

    return dictionary_results



# Set up Langchain summarization with RAG (Retrieval-Augmented Generation)
def summarize_email(sender, date, email_body, current_response, conversation_history, attachments_as_text, dictionary_db, email_history_db):
    model = OllamaLLM(model="llama3.1:latest")
    # Finding incoterms that the model does not understand
    unknown_words = model.invoke("This is an email containing information and/or conversations for a shipping company.\
    I want you to scan through the documents and provide a list of special terms (general shipping terms and incoterms) that you dont understand the meaning of.\
    These terms that you might not be able to understand are like the following: COT, ETC, ISO, SDR\
    The output should be a list in the following format (without any other text): term_a, term_b, term_c, ...")
    
    # Retrieve explanations for specific shipping terms that the model might not recognize
    dictionary_results = retrieve_context_dictionary_db(dictionary_db, unknown_words)
    
    # Combine the context into a single prompt for summarization
    context = "\n\n".join([f"Dictionary Explanation: {doc.page_content}" for doc in dictionary_results])
    # context += "\n\n" + "\n\n".join([f"Previous Email: {doc.page_content}" for doc in email_history_results])
    context += "\n\n" + "\n\n".join([f"Sender: {sender}"])
    context += "\n\n" + "\n\n".join([f"date: {date}"])
    context += "\n\n" + "\n\n".join([f"conversation history: {conversation_history}"])
    context += "\n\n" + "\n\n".join([f"Attached files: {attachments_as_text}"])
    # print(context)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question="Provide a detailed summary of the current email using " \
        "the dictionary explanations to better understand special terms in the email. The summarization " \
        "needs to contain the sender of the email and the date. If the attachments contain shipping numbers " \
        "or any identification numbers whatsoever they need to also be added in the summary. The summary must " \
        "also contain a description of the attached files.")
    
    
    # Load the language model
    response_text = model.invoke(prompt)

    # sources = [doc.metadata.get("id", None) for doc in context]
    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    # formatted_response = f"Response: {response_text}\n"
    # print(formatted_response)
    return response_text
    
def update_history_db(summary):
    
    return

def evaluate_summary(summary, email_txt, attachments_txt):
    model = OllamaLLM(model="llama3.1:latest")
    
    prompt_template = ChatPromptTemplate.from_template(REVIEW_TEMPLATE)
    prompt = prompt_template.format(email_text=email_txt,attachments_text=attachments_txt,summary_text=summary)
    
    review = model.invoke(prompt)
    
    # Specify the file name
    file_name = "review_summary.txt"

    # Open the file in write mode and write new content
    with open(file_name, 'w') as file:
        file.write(review)  
       
    return 
    


from initDictionary import calculate_chunk_ids, split_documents
def add_to_chroma(chunks: list[Document], db):
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
        # db.persist() # happens automatically
    else:
        print("✅ No new documents to add")





def add_summary_to_db(summary, email_history_DB):
    current_time = datetime.now()
    document = Document(page_content=summary, metadata={"source": current_time.strftime("%Y-%m-%d %H:%M:%S")})
    documents = []
    documents.append(document)
    chunks = split_documents(documents)
    add_to_chroma(chunks, email_history_DB)
    # documents.append(document)    
    return







def main():    
    # Load email history and dictionary
    dictionary_DB, email_history_DB = init_DBs()

    sender, date, email_body, current_response, conv_history, attachments_as_text = emailHandler.simple_extract_msg_content("MAILS\\212144_I_ACY-CORR.msg")

    # Generate summary
    start_time = time.time()
    summary = summarize_email(sender, date, email_body, current_response, conv_history, attachments_as_text, dictionary_DB, email_history_DB)
    print("Email Summary:\n", summary)
    add_summary_to_db(summary,email_history_DB)
    summary_time = time.time()
    execution_time = summary_time - start_time
    print(f"Summarization time: {execution_time} seconds")
    
    # Specify the file name
    file_name = "email_summary.txt"

    # Open the file in write mode and write new content
    with open(file_name, 'w') as file:
        file.write(summary)
        
    evaluate_summary(summary, email_body, attachments_as_text)
    review_time = time.time()
    execution_time = review_time - summary_time
    print(f"Review time: {execution_time} seconds")
    
        
    

if __name__ == "__main__":
    main()


