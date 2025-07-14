from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings


from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

#Initialize the model
llm = init_chat_model("gemini-2.5-pro", model_provider="google_genai",
    google_api_key=GOOGLE_API_KEY)



#Embeddings
# embeddings = HuggingFaceEmbeddings(model_name ="all-MiniLM-L6-v2") 
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)




def process_pdf_for_mcqs(pdf_path:str, num_questions : int = 5):
  """Processing PDF and store in vector database for MCQ Generation"""
  from langchain.document_loaders import PyPDFLoader

  loader = PyPDFLoader(pdf_path)
  documents = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size= 1000,
    chunk_overlap =200
  )
  chunks = text_splitter.split_documents(documents)

  #Combine chunks into context (or process individually)
  context = "\n\n".join([chunk.page_content for chunk in chunks[:4]])

  #Generate MCQs using LLM
  mcq_prompt = f"""
  Based o the following content, generate {num_questions} multiple choice questions.
    Each question should have 4 options (A, B, C, D) with one correct answer.

    Content: {context}

    Format each question as:
    Question: [question text]
    A) [option]
    B) [option]
    C) [option]
    D) [option]
    Correct Answer: [letter]"""
  
  response = llm.invoke(mcq_prompt)
  return response.content