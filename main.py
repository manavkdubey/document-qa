import streamlit as st
from transformers import pipeline
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import os
import tempfile

# Load your pre-trained model using Hugging Face pipeline
@st.cache(allow_output_mutation=True)
def load_qa_pipeline():
    return pipeline("question-answering")

qa_pipeline = load_qa_pipeline()

# Helper function to convert PDF to images
def convert_pdf_to_images(pdf_file):
    return convert_from_path(pdf_file)

# Helper function to perform OCR on images
def perform_ocr_on_images(images):
    extracted_text = ""
    for image in images:
        text = pytesseract.image_to_string(image, lang='eng')
        extracted_text += text + "\n\n"  # Separate pages by newlines
    return extracted_text.strip()

# Streamlit app
st.title("Document QA System")

# File uploader widget
uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
        temp_pdf.write(uploaded_file.getvalue())
        temp_pdf_path = temp_pdf.name
    
    # Convert PDF to images
    st.write("Converting PDF pages to images...")
    images = convert_pdf_to_images(temp_pdf_path)
    
    # Perform OCR on images
    st.write("Extracting text from images...")
    manual_text = perform_ocr_on_images(images)
    
    # Clean up temporary PDF file
    os.unlink(temp_pdf_path)
    
    # Display extracted text (optional)
    if st.checkbox("Show extracted text from PDF images"):
        st.text_area("Extracted text:", manual_text, height=300)
    
    # QA Section
    st.header("Ask a question based on the manual")
    question = st.text_input("What do you want to know about the IoT device?")
    if question:
        answer = qa_pipeline(question=question, context=manual_text)
        st.text("Answer:")
        st.write(answer['answer'])
else:
    st.write("Please upload a PDF file.")
