import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
import emailHandler


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

EMB_FUNC = OllamaEmbeddings(model="llama3.1:latest")

# Initialize Chroma DBs for dictionary and email history
def init_DBs():
    dictionary_db = Chroma(persist_directory="Dictionary_DB", embedding_function=EMB_FUNC)
    email_history_db = Chroma(persist_directory="Email_History_DB", embedding_function=EMB_FUNC)
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
    prompt = prompt_template.format(context=context, question="Summarize the current email using \
        the dictionary explanations to better understand special terms in the email. The summarization\
        needs to contain the sender of the email and the date. If the attachments contain shipping numbers\
        or any identification numbers whatsoever they need to also be added in the summary. The summary must\
        also contain a description of the attached files.")
    
    
    # Load the language model
    response_text = model.invoke(prompt)

    # sources = [doc.metadata.get("id", None) for doc in context]
    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    # formatted_response = f"Response: {response_text}\n"
    # print(formatted_response)
    return response_text
    


def main():
    # Load email history and dictionary
    dictionary_DB, email_history_DB = init_DBs()

    sender, date, email_body, current_response, conv_history, attachments_as_text = emailHandler.simple_extract_msg_content("MAILS\\212144_I_ACY-CORR.msg")

    # Generate summary
    summary = summarize_email(sender, date, email_body, current_response, conv_history, attachments_as_text, dictionary_DB, email_history_DB)
    print("Email Summary:\n", summary)

if __name__ == "__main__":
    main()


