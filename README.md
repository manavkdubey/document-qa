


# Document QA System

This Streamlit application is designed to provide a Question-Answering (QA) system for PDF documents. It leverages Optical Character Recognition (OCR) to extract text from uploaded PDF files and uses a pre-trained model from Hugging Face's `transformers` library to answer questions based on the extracted text.

## Features

- **PDF Upload**: Users can upload PDF documents to the application.
- **PDF to Image Conversion**: The application converts PDF pages into images.
- **OCR Processing**: Extracts text from the converted images using `pytesseract`.
- **Question Answering**: Users can ask questions based on the extracted text, and the app uses a pre-trained QA model to provide answers.

## Requirements

To run this application, you need the following libraries:

- `streamlit`: For creating the web application.
- `transformers`: From Hugging Face, used for the QA model.
- `pdf2image`: For converting PDF pages to images.
- `pytesseract`: For OCR capabilities.
- `Pillow`: For image processing.
- `tempfile`: For handling temporary files.
- `os`: For operating system dependent functionalities.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/manavkdubey/document-qa
   ```

2. **Install dependencies:**
   You can install the necessary libraries using pip. It's recommended to do this in a virtual environment:
   ```bash
   pip install streamlit transformers pdf2image pytesseract Pillow
   ```

3. **Install Tesseract-OCR:**
   Pytesseract requires the Tesseract-OCR engine. Follow the installation instructions specific to your operating system.

## Usage

1. **Run the Streamlit app:**
   In the project directory, execute:
   ```bash
   streamlit run app.py
   ```

2. **Upload PDF Document:**
   Use the file uploader to upload a PDF document.

3. **Extract Text:**
   The app automatically converts the PDF to images, performs OCR, and extracts the text.

4. **Ask Questions:**
   Enter your question in the provided text input field. The app will display the answer based on the context of the extracted text.

## Note

- The performance of the OCR depends on the quality of the PDF images.
- The accuracy of answers depends on the pre-trained QA model from Hugging Face.

## Contributions

Contributions are welcome. Please fork the repository and submit pull requests with your enhancements.

## License

This repository is under [MIT License](./LICENSE)



