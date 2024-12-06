from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from halo import Halo

# Load Mistral model and tokenizer from Hugging Face Hub
model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # Update this if using another Mistral variant

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

def review_summary_with_mistral(email_text, email_category, summary):
    prompt = f"""
    You are an expert email reviewer tasked with evaluating the quality of a summary generated for a professional email.

    Please review the following summary based on the original email content and provide:
    1. An evaluation of its accuracy, conciseness, and clarity.
    2. Suggestions for improvement, if any.
    3. A confidence score out of 10 for how well the summary captures the email's intent.

    ### Original Email:
    {email_text}

    ### Attachments (as text):
    {attachments_as_text}

    ### Email Category:
    {email_category}

    ### Summary to Review:
    {summary}

    Output Format:
    1. Accuracy: [High/Medium/Low]
    2. Suggestions: [Improvement points]
    3. Confidence Score: [Score out of 10]
    """
    spinner = Halo(text='Reviewing Summary...', spinner='dots')
    spinner.start()
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate a response
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300, pad_token_id=tokenizer.eos_token_id)
    spinner.stop()
    # Decode and return the output
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example Usage
email_text = "The client needs the project completed by Friday."
attachments_as_text = "No attachments."
email_category = "Logistics"
summary = "The client requires the project to be completed by Friday."


review_result = review_summary_with_mistral(email_text, email_category, summary)
print(review_result)
