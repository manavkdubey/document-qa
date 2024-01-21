from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(input_text):
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

import streamlit as st

st.title("Chatbot with Hugging Face and LangChain")

user_input = st.text_input("You:", "")
if user_input:
    bot_response = generate_response(user_input)
    st.text_area("Bot:", value=bot_response, height=200)
