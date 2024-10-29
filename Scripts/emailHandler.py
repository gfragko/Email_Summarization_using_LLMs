import extract_msg
import pytesseract
from PIL import Image
import io
import os
import fitz  # PyMuPDF for PDF processing
from docx import Document  # python-docx for DOCX processing
import olefile  # for older .doc files (requires pywin32 on Windows)
import re
import PyPDF2
from pdf2image import convert_from_bytes


    
def simple_extract_msg_content(msg_path):
    contents = ''
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
            text = process_image_attachment(attachment)
            attachments_as_text.append(text)
            print("Found image")

        # Check if the attachment is a PDF
        elif filename.endswith('.pdf'):
            # process_pdf_attachment(attachment)
            text = process_pdf_attachment(attachment)
            attachments_as_text.append(text)
            print("Found PDF")

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


def process_image_attachment(attachment):
    image_data = attachment.data  # Binary data of the image
        
    # Convert binary image data to an Image object using PIL
    image = Image.open(io.BytesIO(image_data))
        
    # Use Tesseract to extract text from the image
    extracted_text = pytesseract.image_to_string(image)
    
    
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
        extracted_text = extract_text_from_image(pdf_data)  
    else:
        print("The string contains visible characters.")  
    
    
    return extracted_text



def extract_text_from_image(pdf_data):
    content = ''
    images = convert_from_bytes(pdf_data)
    for page_num, image in enumerate(images):
        # Optionally save each page as an image file (e.g., for reference)
        # image.save(f"page_{page_num + 1}.jpg", "JPEG")
        
        # Alternatively, use Tesseract to extract text from the image directly
        extracted_text = pytesseract.image_to_string(image)
        
        # Print the extracted text from the image
        content += extracted_text
    
    return content 








def main():
    current_response, conversation_history, attachments_as_text = simple_extract_msg_content("C:\\Users\\gfrag\\Desktop\\Workspace\\MAILS\\212144_I_ACY-CORR.msg")




    print("=============================================================")
    print(f"The current mail is: \n{current_response}")
    print("=============================================================")
    print("The conversation history is: \n")
    for mail in conversation_history:
        print(mail,"\n")
    print("=============================================================")
    print(f"The attachments are: \n")
    i = 1
    for att in attachments_as_text:
        print(i,".->>>>>>>>>>",att,"\n")
    print("=============================================================")
    
if __name__ == "__main__":
    main()