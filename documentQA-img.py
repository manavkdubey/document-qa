import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import tempfile
import torch
from transformers import LayoutLMForQuestionAnswering, LayoutLMTokenizer


try:
    model_name = "microsoft/layoutlmv3-large"  # Ensure this is a valid model checkpoint
    tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlmv3-large")
    model = LayoutLMForQuestionAnswering.from_pretrained("microsoft/layoutlmv3-large")
except Exception as e:
    print(f"An error occurred: {e}")

# Function to perform OCR and convert image to model's input format
def process_image_to_model_input(image):
    # Perform OCR on the image to get bounding boxes
    ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
    words = list(ocr_df[ocr_df.conf != -1].text)
    coordinates = ocr_df[ocr_df.conf != -1][['left', 'top', 'width', 'height']].values
    # Normalize the coordinates
    width, height = image.size
    actual_boxes = []
    for (left, top, width, height) in coordinates:
        actual_boxes.append((left, top, left + width, top + height))
    return words, actual_boxes

# Streamlit app interface
st.title("Document Question Answering with LayoutLM")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
        temp_pdf.write(uploaded_file.getvalue())
        temp_pdf_path = temp_pdf.name

    # Convert first page of PDF to image
    images = convert_from_path(temp_pdf_path, first_page=0, last_page=1)
    image = images[0]
    st.image(image, caption='Uploaded PDF Image', use_column_width=True)

    # Process the image
    words, boxes = process_image_to_model_input(image)
    
    # Input for LayoutLM is different as it needs token_type_ids and attention_mask along with input_ids
    encoding = tokenizer(words, boxes=boxes, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    token_type_ids = encoding["token_type_ids"]

    # Ask a question
    question = st.text_input("What would you like to know from this document?")
    if question:
        # Encode the question with the context of the words and boxes
        question_encoding = tokenizer(question, return_tensors="pt")
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }
        # Merge the question encoding with the context encoding
        inputs.update({k: torch.cat([v, question_encoding[k]]) for k in question_encoding})
        
        # Get the model's answer
        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        
        # Get the most probable start and end of answer indices
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        # Convert indices to tokens
        answer_tokens = input_ids[0, answer_start:answer_end]
        answer = tokenizer.decode(answer_tokens)
        
        # Display the answer
        st.write(answer)
else:
    st.write("Please upload a PDF file.")
