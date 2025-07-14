import streamlit as st
from llm_wrapper import process_pdf_for_mcqs
import tempfile


st.write("MCQ Generator")
uploaded_docs  = st.file_uploader("Upload your document.", type = ["pdf"])

if uploaded_docs is not None:
  # Save uploaded file to temporary loc
  with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
    tmp_file.write(uploaded_docs.read())
    tmp_path = tmp_file.name


  llm_response = process_pdf_for_mcqs(tmp_path)
  st.write(llm_response)
