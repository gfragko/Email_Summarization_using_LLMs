import time
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain.schema.document import Document
from datetime import datetime
import emailHandler
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Initialize the SentenceTransformer model (same model used during database creation)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define a function to generate embeddings (same as during database creation)
def embed_text(texts):
    return embedding_model.encode(texts, convert_to_numpy=True)


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

# EMB_FUNC = OllamaEmbeddings(model="llama3.1:latest")

# Initialize Chroma DBs for dictionary and email history
def get_DBs():
    dictionary_db = Chroma(persist_directory="Dictionary_DB", embedding_function=embed_text)
    return dictionary_db



# Function to retrieve context from both Chroma DBs
def retrieve_incoterms_context(dictionary_db, query):
    # Retrieve the meaning of special terms from the dictionary
    print("finding incoterms relevance...")
    dictionary_results = dictionary_db.similarity_search(query, k=5)
    print("found")
    return dictionary_results




# Set up Langchain summarization with RAG (Retrieval-Augmented Generation)
def summarize_email(sender, date, email_body, current_response, conversation_history, attachments_as_text, dictionary_db, email_history_db):
    # model = OllamaLLM(model="llama3.1:latest")
    model = OllamaLLM(model="llama3.2-vision")
    prompt = f"[EMAIL]: {conversation_history} \n\n This is an email containing information and/or conversations for a shipping company."\
    "I want you to scan through the documents and provide a list of special terms (general shipping terms and incoterms) that you dont understand the meaning of."\
    "These terms that you might not be able to understand could be like but not limited to the following: COT, ETC, ISO, SDR(THESE ARE EXAMPLES that might not exist in the email). "\
    "The output should be a list in the following format (without any other text): term_a, term_b, term_c, ..."
    print("Unknown words prompt:", prompt)
    # Finding incoterms that the model does not understand
    unknown_words = model.invoke(prompt)
    
    print("Unknown words:", unknown_words)
    #####################################################################################
    prompt = f"""
    [EMAIL]: {conversation_history} \n\n This is an email containing information and/or conversations for a shipping company.
    I want you to turn this text into a more clear version of the conversation that is provided which contains ONLY the main body of each text email message that was exchanged.
    The output you will provide should use the following formatting:
    [Sender user 1]: [main body of email they sent]
    [Sender user 2]: [main body of email they sent]
    Sender usernames must be extracted from the email text i have provided.
    For example the following message:
    From: Skysealand <ssls@colbd.com>
    Sent: Πέμπτη, 19 Αυγούστου 2021 2:26 μμ
    To: Lia Charalampopoulou <operations@arianmaritime.gr>
    Cc: Vicky Parissi <operations01@arianmaritime.gr>; Marina Koletzaki <mkoletzaki@arianmaritime.gr>
    Subject: RE: 212144 * SSL21169
    Dear Sir,
    Pls find attached of Surrendered MB Copy\nB.Regards.
    AMDAD HOSSAN  SR. EXECUTIVE
    EXPORT DOCUMENTION OF
    SKYSEALAND SHIPPING LINES
    TEL:+88031-2526344 FAX:+88031-2523955
    MOBILE NO.+8801840-867611
    EMAIL#ssls@colbd.com
    +++++++++++++++++++++++++++++
    The formatted form of this message should be:
    Skysealand <ssls@colbd.com> : Dear Sir,
    Pls find attached of Surrendered MB Copy
    B.Regards.
    """
    clean_conversation = model.invoke(prompt)
    
    print("Formatted conversation",clean_conversation)
    
    
    #####################################################################################
    
    attached_mentioned_flag = model.invoke(f"{clean_conversation}\n\nLook through this conversation and tell me if the people that communicate are looking for information from an attached file.The output you will provide must be in the following format without further comments:(If true)->  TRUE, [information they are searching for] \n(If false)-> FALSE")
    
    print(attached_mentioned_flag)
    
    
    
    # Retrieve explanations for specific shipping terms that the model might not recognize
    dictionary_results = retrieve_context_dictionary_db(dictionary_db, unknown_words)
    
    # Combine the context into a single prompt for summarization
    context = "\n\n".join([f"Dictionary Explanation: {doc.page_content}" for doc in dictionary_results])
    # context += "\n\n" + "\n\n".join([f"Previous Email: {doc.page_content}" for doc in email_history_results])
    context += "\n\n" + "\n\n".join([f"Sender: {sender}"])
    context += "\n\n" + "\n\n".join([f"date: {date}"])
    context += "\n\n" + "\n\n".join([f"conversation history: {clean_conversation}"])
    flags = ["false", "FALSE", "False"]
    if all(flag.lower() not in attached_mentioned_flag.lower() for flag in flags):
        print("!!!!!!!!!!!!!!!!!ATTACHED FILES INCLUDED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        context += "\n\n" + "\n\n".join([f"Attached files: {attachments_as_text}"])
    # print(context)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question="Provide a detailed summary of the current email using " \
        "the dictionary explanations to better understand special terms in the email. The summarization " \
        "needs to contain the sender of the email and the date. If the attachments contain shipping numbers " \
        "or any identification numbers whatsoever they need to also be added in the summary. The summary must " \
        "also contain a description of the attached files."\
        "These contracts are signed off by both parties involved (the shipper/exporter and the consignee/importer) and you are allowed to edit them")
    
    
    # Load the language model
    response_text = model.invoke(prompt)

    # sources = [doc.metadata.get("id", None) for doc in context]
    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    # formatted_response = f"Response: {response_text}\n"
    # print(formatted_response)
    return response_text
    


def evaluate_summary(summary, email_txt, attachments_txt):
    model = OllamaLLM(model="llama3.1:latest")
    
    prompt_template = ChatPromptTemplate.from_template(REVIEW_TEMPLATE)
    prompt = prompt_template.format(email_text=email_txt,attachments_text=attachments_txt,summary_text=summary)
    
    review = model.invoke(prompt)
    
    # Specify the file name
    file_name = "Results\review_summary.md"

    # Open the file in write mode and write new content
    with open(file_name, 'w') as file:
        file.write(review)  
       
    return 
    



def main():    
    # Load email history and dictionary
    dictionary_DB = get_DBs()

    sender, date, email_body, current_response, conv_history, attachments_as_text = emailHandler.simple_extract_msg_content("MAILS\\212144_I_ACY-CORR.msg")

    # Generate summary
    start_time = time.time()
    summary = summarize_email(sender, date, email_message, attachments_as_files, dictionary_DB, email_category)
    # print("Email Summary:\n", summary)
    summary_time = time.time()
    execution_time = summary_time - start_time
    print(f"Summarization time: {execution_time} seconds")
    
    # Specify the file name
    file_name = "Results\email_summary.md"

    # Open the file in write mode and write new content
    with open(file_name, 'w') as file:
        file.write(summary)
        
    evaluate_summary(summary, email_body, attachments_as_text)
    review_time = time.time()
    execution_time = review_time - summary_time
    print(f"Review time: {execution_time} seconds")
    
        
    

if __name__ == "__main__":
    main()


