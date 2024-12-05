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
from langchain_huggingface import HuggingFaceEmbeddings
from categorize_email import categorize_email

# Initialize the SentenceTransformer model (same model used during database creation)
embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')



# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """

SUMMARY_TEMPLATE = """
You are a highly skilled assistant summarizing professional emails for streamlined communication. Analyze the following email and attachments to create a summary that highlights key information, actions required, and any important deadlines.

Details:

Sender: {sender}
Date: {date}
Email Category: {email_category}
Email Content:
{text}

Attachments (as text):
{attachments_as_text}

Explanations for unknown words:
{db_context}


Summary Instructions:
Identify the main purpose of the email and summarize it in one or two sentences.
Highlight any key points, requests, or deadlines mentioned in the email or attachments.
Specify any required actions for the recipient.
If applicable, include relevant context based on the email category.
Output Example:

Main Purpose: [Brief summary of the email’s overall intent]
Key Points/Requests:
[Key point 1]
[Key point 2]
Action Items:
[Action required, responsible party, and deadline if applicable]
Ensure the summary is clear, concise, and reflects the email's intent accurately.


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
    dictionary_db = Chroma(persist_directory="Dictionary_DB", embedding_function=embedding_function)
    return dictionary_db



# Function to retrieve context from both Chroma DBs
def retrieve_incoterms_context(dictionary_db, query):
    # Retrieve the meaning of special terms from the dictionary
    print("finding incoterms relevance...")
    dictionary_results = dictionary_db.similarity_search(query, k=5)
    print("found")
    return dictionary_results




# Set up Langchain summarization with RAG (Retrieval-Augmented Generation)
def summarize_email(sender, date, current_response, attachments_as_text, dictionary_db, email_category):
    # model = OllamaLLM(model="llama3.1:latest")
    model = OllamaLLM(model="llama3.2-vision:latest")
    prompt = f"[EMAIL]: {current_response} \n\n This is an email sent to a shipping company."\
    "I want you to scan through the given EMAIL text above and provide a list of special terms (general shipping terms and incoterms) that you dont understand the meaning of."\
    "These terms that you might not be able to understand could be like but not limited to the following: COT, ETC, ISO, SDR(THESE ARE EXAMPLES that might not exist in the email). "\
    "The output should be a list in the following format (without any other text): term_a, term_b, term_c, ..."
    print("Unknown words prompt:", prompt)
    # Finding incoterms that the model does not understand
    unknown_words = model.invoke(prompt)
    
    print("Unknown words:", unknown_words)
    
    
    # Retrieve explanations for specific shipping terms that the model might not recognize
    dictionary_results = retrieve_incoterms_context(dictionary_db, unknown_words)
    print("dictionary_results:", unknown_words)
    
    
    
    # Combine the context into a single prompt for summarization
    context = "\n\n".join([f"Dictionary Explanation: {doc.page_content}" for doc in dictionary_results])
    # context += "\n\n" + "\n\n".join([f"Previous Email: {doc.page_content}" for doc in email_history_results])
    # context += "\n\n" + "\n\n".join([f"Sender: {sender}"])
    # context += "\n\n" + "\n\n".join([f"date: {date}"])
    # context += "\n\n" + "\n\n".join([f"Email text: {current_response}"])
    # context += "\n\n" + "\n\n".join([f"Attached files: {attachments_as_text}"])

    # prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # prompt = prompt_template.format(context=context, question="Provide a detailed summary of the current email using " \
    #     "the dictionary explanations to better understand special terms in the email. The summarization " \
    #     "needs to contain the sender of the email and the date. If the attachments contain shipping numbers " \
    #     "or any identification numbers whatsoever they need to also be added in the summary. The summary must " \
    #     "also contain a description of the attached files."\
    #     "These contracts are signed off by both parties involved (the shipper/exporter and the consignee/importer) and you are allowed to edit them")
    prompt_template = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)
    prompt = prompt_template.format(sender=sender, date=date, email_category=email_category,
                                    text=current_response, attachments_as_text=attachments_as_text, db_context=context)
    
    
    # Load the language model
    response_text = model.invoke(prompt)
    
    return response_text
    


# def evaluate_summary(summary, email_txt, attachments_txt):
#     model = OllamaLLM(model="llama3.2-vision:latest")
    
#     prompt_template = ChatPromptTemplate.from_template(REVIEW_TEMPLATE)
#     prompt = prompt_template.format(email_text=email_txt,attachments_text=attachments_txt,summary_text=summary)
    
#     review = model.invoke(prompt)
    
#     # Specify the file name
#     file_name = "Results\review_summary.md"

#     # Open the file in write mode and write new content
#     with open(file_name, 'w') as file:
#         file.write(review)  
       
#     return 
    


def main():    
    # Load email history and dictionary
    dictionary_DB = get_DBs()

    sender, date, email_body, current_response, conv_history, attachments_as_text = emailHandler.simple_extract_msg_content("MAILS\\212144_I_ACY-CORR.msg")

    # Generate summary
    start_time = time.time()
    
    email_category = categorize_email(current_response)
    summary = summarize_email(sender, date, current_response, attachments_as_text, dictionary_DB, email_category)
    summary_time = time.time()
    execution_time = summary_time - start_time
    print(f"Summarization time: {execution_time} seconds")
    
    # Specify the file name
    file_name = "Results\email_summary.md"

    # Open the file in write mode and write new content
    with open(file_name, 'w') as file:
        file.write(summary)
    
    # from evaluate_summary import review_summary_with_mistral
    # review = review_summary_with_mistral(current_response, attachments_as_text, email_category, summary)
    # # Specify the file name
    # file_name = "Results\review.md"

    # # Open the file in write mode and write new content
    # with open(file_name, 'w') as file:
    #     file.write(review)
        
    # review_time = time.time()
    # execution_time = review_time - summary_time
    # print(f"Review time: {execution_time} seconds")
    
        
    

if __name__ == "__main__":
    main()


