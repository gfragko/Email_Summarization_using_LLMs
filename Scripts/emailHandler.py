import extract_msg
import pytesseract
from PIL import Image
import io
import re
import PyPDF2
from pdf2image import convert_from_bytes
import ollama
from tempfile import mkdtemp
import os
import shutil
from pdf2image import convert_from_path
# import fitz  # PyMuPDF for PDF processing
# from docx import Document  # python-docx for DOCX processing
# import olefile  # for older .doc files (requires pywin32 on Windows)


    
def simple_extract_msg_content(msg_path):
    """
     This returns: + sender, date, email_body(all of the text in the msg file without any formatting),
                   + current_response(last message in the conversation), 
                   + conversation_history (an array where each element is an email message in the conversation, before the current response), 
                   + attachments_as_text (an array where each element is the extracted text from an attached file)
    """
    msg = extract_msg.Message(msg_path)
    email_body = msg.body
    sender = msg.sender
    date = msg.date
    
    # Example of splitting based on quoted responses or common reply markers
    if "From:" in email_body:
        # responses = email_body.split("From:")
        responses = re.split(r'(?=From:)', email_body)
        current_response = responses[0]  # The current email content
        conversation_history = responses[1:]
        
    
    attachments = msg.attachments
    attachments_as_text = []
    for attachment in attachments:
        
        filename = attachment.longFilename.lower()
               

        # Check if the attachment is an image
        if filename.endswith(('png', 'jpg', 'jpeg')):
            print("Found image")
            text = process_image_with_vision(attachment)
            attachments_as_text.append(text)
            

        # Check if the attachment is a PDF
        elif filename.endswith('.pdf'):
            # process_pdf_attachment(attachment)
            print("Found PDF")
            text = process_pdf_attachment(attachment)
            attachments_as_text.append(text)
            

        # Check if the attachment is a Word document
        elif filename.endswith('.docx'):
            # process_docx_attachment(attachment)
            print("Found docx")

        elif filename.endswith('.doc'):
            # process_doc_attachment(attachment)
            print("Found doc")
            
        else:
            print("unused document found")
    
    return sender, date, email_body, current_response, conversation_history, attachments_as_text

#depricated
# def process_image_attachment(attachment):
#     image_data = attachment.data  # Binary data of the image
        
#     # Convert binary image data to an Image object using PIL
#     image = Image.open(io.BytesIO(image_data))
        
#     # Use Tesseract to extract text from the image
#     extracted_text = pytesseract.image_to_string(image)
#     # print("--------------------------------------------------------")
#     # print(extracted_text)
#     # print("--------------------------------------------------------")
    
    
#     return extracted_text

def extract_text_from_path(path):
    prompt = """This is a scanned document. 
    You must perform high level ocr on this document in order to extract the text it contains.
    The text extraction must be content and layout aware, since the document might contain tables
    Your output should be the ONLY the extracted text, without any extra comments."""

    response = ollama.chat(
        model='llama3.2-vision',
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [path]
        }]
    )     
       
    extracted_text = response['message']['content']    
    return extracted_text
    
    
def process_image_with_vision(attachment):
    # Create a temporary directory to store the attachments
    temp_dir = mkdtemp()   
    try:
        file_name = attachment.longFilename
        file_path = os.path.join(temp_dir, file_name)
        
        # Save the attachment to the temporary directory
        with open(file_path, 'wb') as f:
            f.write(attachment.data)

        # Process the file
        print(file_path)
        extracted_text = extract_text_from_path(file_path)

    finally:
        # Delete the temporary directory and all its contents
        shutil.rmtree(temp_dir)
        print(f"Temporary directory deleted: {temp_dir}")    
    return extracted_text




def process_scanned_pdf(pdf_path):
    """
    Converts each page of a scanned PDF into an image, extracts text from each image,
    combines the text, and deletes the temporary images.
    """
    # Create a temporary directory to store the images
    temp_dir = mkdtemp()
    print(f"Temporary directory created: {temp_dir}")

    try:
        # Convert PDF pages to images and save them in the temporary directory
        images = convert_from_path(pdf_path, output_folder=temp_dir, fmt='jpg')

        # Initialize an empty string to hold the combined text
        combined_text = ""

        # Extract text from each image and combine the results
        for i, image in enumerate(images):
            image_path = os.path.join(temp_dir, f"page_{i+1}.jpg")
            image.save(image_path, 'JPG')

            # Extract text from the image
            extracted_text = extract_text_from_path(image_path)

            # Append the extracted text to the combined text
            combined_text += extracted_text + "\n"

        return combined_text

    finally:
        # Delete the temporary directory and all its contents
        shutil.rmtree(temp_dir)
        print(f"Temporary directory deleted: {temp_dir}")




def extract_text_from_scanned_pdf(attachment):
    temp_dir = mkdtemp()
    
    try:
        file_name = attachment.longFilename
        if file_name.endswith('.pdf'):  # Ensure the attachment is a PDF
            pdf_path = os.path.join(temp_dir, file_name)
                
            # Save the PDF attachment temporarily
            with open(pdf_path, 'wb') as f:
                f.write(attachment.data)
                
            # Process the extracted PDF
            extracted_text = process_scanned_pdf(pdf_path)

    finally:
        # Clean up by deleting the temporary directory and all its contents
        shutil.rmtree(temp_dir)
        print(f"Temporary directory deleted: {temp_dir}")
    
    
    
    return extracted_text  


def process_pdf_attachment(attachment):
    extracted_text = ''
    
    pdf_data = attachment.data
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
    
    # Loop through all the pages of the PDF
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        extracted_text += page.extract_text()
    
    
    print("trying to find text...")   
    if not extracted_text.strip():
        print("Pdf is a scanned document...")        
        extracted_text = extract_text_from_scanned_pdf(attachment)  
    else:
        print("The string contains visible characters.")  
    
    
    return extracted_text


# Depricated
# def extract_text_from_image(pdf_data):
#     content = ''
#     images = convert_from_bytes(pdf_data)
#     for page_num, image in enumerate(images):
#         # Optionally save each page as an image file (e.g., for reference)
#         # image.save(f"page_{page_num + 1}.jpg", "JPEG")
        
#         # Alternatively, use Tesseract to extract text from the image directly
#         extracted_text = pytesseract.image_to_string(image)
        
#         # Print the extracted text from the image
#         content += extracted_text
    
#     return content 






# Main function for testing
def main():
    sender, date, email_body,current_response, conversation_history, attachments_as_text = simple_extract_msg_content("MAILS\\212144_I_ACY-CORR.msg")
    # sender, date, email_body, current_response, conversation_history, attachments_as_text



    print("=============================================================")
    # print(f"The current mail is: \n{current_response}")
    print("=============================================================")
    print("The conversation history is: \n")
    indx = 0
    for mail in conversation_history:
        indx +=1
        print(f"{indx}.->\n {mail}\n=================================")
    print("=============================================================")
    
    print(conversation_history[2])
    print(f"The attachments are: \n")
    i = 1
    for att in attachments_as_text:
        print(i,".->>>>>>>>>>",att,"\n")
        i += 1
    print("=============================================================")
    
if __name__ == "__main__":
    main()