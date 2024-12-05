from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

model = OllamaLLM(model="llama3.2-vision")

oneshot_template="""
As an expert email classifier, you are equipped to analyze and categorize any given email into one of the following categories:

+ Update for cargo transportation
+ Advertisement
+ Spam/Phishing
+ Request for Quote
+ Offer
+ Accepted or Not
+ Booking Note
+ Reference Number Execution

Your output must be ONLY the name of the category.

Below are examples of each category for reference:

+ Update for Cargo Transportation ->
Subject: Update on Cargo Shipment Schedule
Body:
Dear [Recipient's Name],
We would like to inform you that the shipment for Order #456789 has been delayed due to unforeseen weather conditions. The new estimated time of arrival at Port XYZ is November 28, 2024.

We apologize for any inconvenience caused and will keep you updated on further developments.

Best regards,
[Your Name]
[Your Position]
[Company Name]

+ Advertisement ->
Subject: Save Big on Our Logistics Services!
Body:
Hi [Recipient's Name],
Unlock unbeatable rates for your cargo transportation needs! For a limited time, we're offering up to 30% off on international freight services.

Don't miss this opportunity—request a quote today and experience reliable, efficient shipping at the best prices!

Best wishes,
[Your Name]
[Marketing Team]
[Company Name]

+ Spam/Phishing ->
Subject: URGENT: Update Your Account Information
Body:
Dear Customer,
Your account has been flagged for unusual activity. To avoid suspension, please verify your details immediately by clicking the link below:
[Click Here to Verify]

Failure to act within 48 hours will result in account suspension.

Thank you,
Support Team
(Note: This is a phishing example—avoid clicking suspicious links!)

+ Request for Quote ->
Subject: Request for Quotation: Bulk Cargo Transportation
Body:
Dear [Recipient's Name],
We are looking for reliable transportation services for a bulk shipment of [cargo type]. Please provide a quotation for the following details:

Volume: [Details]
Pickup Location: [Address]
Delivery Location: [Address]
Timeline: [Required Dates]
Looking forward to your response.

Best regards,
[Your Name]
[Your Position]
[Company Name]

+ Offer ->
Subject: Special Offer: Freight Services at Discounted Rates
Body:
Dear [Recipient's Name],
We are pleased to offer you a 20% discount on all shipments booked by December 15, 2024. This exclusive deal applies to our full range of cargo solutions.

Let us know how we can assist with your upcoming shipments!

Kind regards,
[Your Name]
[Your Position]
[Company Name]

+ Accepted or Not ->
Subject: Confirmation of Quotation Acceptance
Body:
Dear [Recipient's Name],
We are pleased to confirm that your quotation for [service or cargo] has been accepted. Please proceed with the necessary arrangements.

Details of the shipment are as follows:

Reference Number: #789456
Delivery Deadline: [Date]
Thank you for your cooperation!

Best regards,
[Your Name]
[Your Position]
[Company Name]

+ Booking Note ->
Subject: Booking Note: Order #789123
Body:
Dear [Recipient's Name],
We have successfully booked your shipment with the following details:

Booking Reference: BN#123456
Cargo Type: [Details]
Estimated Pickup Date: [Date]
Delivery Location: [Address]
Please let us know if any modifications are required.

Best regards,
[Your Name]
[Your Position]
[Company Name]

+ Reference Number Execution ->
Subject: Execution of Shipment: Reference #789123
Body:
Dear [Recipient's Name],
The shipment associated with Reference Number #789123 has been successfully dispatched. Please find the attached documentation, including the bill of lading and tracking details.

If you have any questions, feel free to contact us.

Kind regards,
[Your Name]
[Your Position]
[Company Name]

Now, analyze the following email and classify it:

Email:
{email_body}


"""





prompt = PromptTemplate.from_template(oneshot_template)
chain_zeroshot = prompt | model


def categorize_email(email_text):
    
    category = chain_zeroshot.invoke({"email_body": email_text})
    
    return category


# print("Spam/Phishing email-> \n",chain_zeroshot.invoke({"email_body": phishing_example}))
# print("##########################################################")
# print("Request for Quote email (real example)->\n",chain_zeroshot.invoke({"email_body": real_example}))