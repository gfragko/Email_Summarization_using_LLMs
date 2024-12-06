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
from halo import Halo

# Initialize the SentenceTransformer model (same model used during database creation)
embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')



SUMMARY_TEMPLATE = """
You are a highly skilled assistant summarizing professional emails for streamlined communication. 
Analyze the following email to create a summary that highlights key information, actions required, and any important deadlines. Use attachments only if additional context or clarification is required.


Details:

+Sender: {sender}
+Date: {date}
+Email Category: {email_category}
+Email Text:
{text}


+Explanations for unknown words:
{db_context}


Summary Instructions:
Identify the main purpose of the Email Text and summarize it in one or two sentences, using the Explanations for unknown words for anything you dont understand.
Highlight any key points, requests, or deadlines mentioned in the email or attachments.
Specify any required actions for the recipient.
If applicable, include relevant context based on the email category.



================================================================================
Output Example:

Main Purpose: [Brief summary of the email’s overall intent]
Key Points/Requests:
[Key point 1]
[Key point 2]
Action Items:
[Action required, responsible party, and deadline if applicable]
Ensure the summary is clear, concise, and reflects the email's intent accurately.


"""


# REFINED_SUMMARY_TEMPLATE = """
# Original Summary:{summary}

# Using the original summary try to see if you can refine the summarization by using context from the following attached files from the email.
# If you cannot achieve something better the output must be the exact text of the original summary.


# Attached files for context:
# {attachments_as_text}
# """

REFINED_SUMMARY_TEMPLATE = """
You are an expert assistant skilled in analyzing and refining email summaries. 
Your task is to assess an original email summary and determine whether the attached files provide additional context that can improve or enhance the summary. 
If no meaningful improvement can be achieved using the context from the attachments, return the original summary exactly as it is, without modification.

Instructions:
Analyze the original summary to understand the main points.
Review the attachments for any additional details, clarifications, or context that could refine or improve the summary.
Only update the original summary if the context from the attachments adds significant value or clarity.
Input:
Original Summary:
{summary}

Attachments (as text):
{attachments_as_text}

Output Format:
If improved: Provide the refined summary incorporating additional context from the attachments.
If no improvement: Return the original summary verbatim, with no changes.
Example Output:
Refined Summary (if applicable):
[Enhanced summary incorporating additional context from attachments.]

OR

Original Summary (unchanged):
[The exact text of the original summary.]
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
    
    # Finding incoterms that the model does not understand
    unknown_words = model.invoke(prompt)
    
    print("Unknown words:", unknown_words)
    
    
    # Retrieve explanations for specific shipping terms that the model might not recognize
    dictionary_results = retrieve_incoterms_context(dictionary_db, unknown_words)   
    
    # Combine the context into a single prompt for summarization
    context = "\n\n".join([f"Dictionary Explanation: {doc.page_content}" for doc in dictionary_results])
    

    prompt_template = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)
    prompt = prompt_template.format(sender=sender, date=date, email_category=email_category, 
                                    text=current_response, db_context=context)
    
    # Load the language model
    print("processing")
    
    spinner = Halo(text='First Summary Draft...', spinner='dots')
    spinner.start()
    summary = model.invoke(prompt)
    spinner.stop()
    
    
    
    prompt_template = ChatPromptTemplate.from_template(REFINED_SUMMARY_TEMPLATE)
    prompt = prompt_template.format(summary=summary,attachments_as_text=" ".join(attachments_as_text))
    spinner = Halo(text='Refining the summary using attached files...', spinner='dots')
    spinner.start()
    refined_summary = model.invoke(prompt)
    spinner.stop()
    return refined_summary
    





def summarize_attachments(attachments_as_text):
    summarized_attachments = []
    prompt_template= """
    Summarize the following document for a shipping company. Maintain all critical details such as shipment numbers, vessel names,
    cargo descriptions, dates, port locations, contractual obligations, and any instructions or actions required. Ensure the summary
    is clear, concise, and easy to understand, but does not omit any important operational, financial, or legal information. Organize 
    the summary logically, grouping related information under appropriate headings if necessary.
    
    Document to summarize:
    {document_content}
    
    """    
    model = OllamaLLM(model="llama3.2-vision:latest")
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | model
    spinner = Halo(text='Summarizing attachments...', spinner='dots')
    spinner.start()
    for attachment in attachments_as_text:
        summary = chain.invoke({"document_content": attachment})
        summarized_attachments.append(summary)
    spinner.stop()  
    return summarized_attachments





def main():    
    # Load email history and dictionary
    dictionary_DB = get_DBs()

    sender, date, email_body, current_response, conv_history, attachments_as_text = emailHandler.simple_extract_msg_content("MAILS/212144_I_ACY-CORR.msg")

    # Generate summary
    start_time = time.time()
    email_text = conv_history[3]
    email_category = categorize_email(email_text)
    summarized_attachments = summarize_attachments(attachments_as_text)
    summary = summarize_email(sender, date, email_text, summarized_attachments, dictionary_DB, email_category)
    print("####################################################################################")
    print("Summary:")
    print(summary)
    print("#################################################################################### \n Original email message: \n", email_text)
    summary_time = time.time()
    execution_time = summary_time - start_time
    print(f"Summarization time: {execution_time} seconds")
    
    # Specify the file name
    file_name = "Results/email_summary_new.md"

    # Open the file in write mode and write new content
    with open(file_name, 'w') as file:
        file.write(summary)
    
    from evaluate_summary import review_summary_with_mistral
    review = review_summary_with_mistral(current_response, email_category, summary)
    
    print("#################################################################################### \n Review: \n", review)
    
    # Specify the file name
    file_name = "Results/review_new.md"

    # Open the file in write mode and write new content
    with open(file_name, 'w') as file:
        file.write(review)
        
    review_time = time.time()
    execution_time = review_time - summary_time
    print(f"Review time: {execution_time} seconds")
    
        
    

if __name__ == "__main__":
    main()


