import os
from dotenv import load_dotenv
import PyPDF2
import pandas as pd

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.chat import ChatPromptTemplate

load_dotenv()
api_key = os.getenv("API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key

chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", temperature=0.7, google_api_key=api_key
)

chat_template = ChatPromptTemplate.from_messages([
    ("system", """
        You are a chatbot. Based on the category of the user's query, respond appropriately.
        If the question is:
        - Technical → Provide detailed explanation
        - Personal → Be polite and friendly
        - General knowledge → Give concise facts
    """),
    ("human", "{text}")
])

def pdf_to_text(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def text_to_text(txt_file):
    return txt_file.read().decode("utf-8")

def excel_to_text(excel_file):
    df = pd.read_excel(excel_file)
    return df.to_string(index=False)

def setup_query_engine(doc_text):
    doc = Document(text=doc_text)
    Settings.llm = Gemini(model_name="models/gemini-1.5-flash")
    Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")
    index = VectorStoreIndex([doc])
    retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
    return RetrieverQueryEngine(retriever=retriever)

def is_rag_query(prompt):
    rag_keywords = [
        "based on the document", "from the file", "according to the report",
        "as per the pdf", "from uploaded", "in the document", "mentioned in the file"
    ]
    return any(kw in prompt.lower() for kw in rag_keywords)

def handle_chat(prompt, query_engine):
    if query_engine and is_rag_query(prompt):
        response = query_engine.query(prompt)
        return str(response)
    else:
        messages = chat_template.format_messages(text=prompt)
        result = chat_model.invoke(messages)
        return result.content
