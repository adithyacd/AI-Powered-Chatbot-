from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,Document
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

import PyPDF2

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate

from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("API_KEY")


chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7,google_api_key=api_key)
template = """
            You are a chatbot. Based on the category of the user's query, respond appropriately.
            If the question is:
                - Technical → Provide detailed explanation
                - Personal → Be polite and friendly
                - General knowledge → Give concise facts
"""
human_template = "{text}"
chat_prompt = ChatPromptTemplate.from_messages([("system",template),("human",human_template)])



def pdf_to_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() 
    return text
doc = pdf_to_text(r"E:\GenAI\temp\Adithya CD Resume.pdf")
docs = Document(text=doc)
#doc = SimpleDirectoryReader(r"E:\GenAI\temp").load_data()
#print(doc)
os.environ["GOOGLE_API_KEY"] = api_key
gemini_llm = Gemini(model_name="models/gemini-1.5-flash")
gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")
Settings.llm = gemini_llm
Settings.embed_model = gemini_embed_model
#index = VectorStoreIndex(doc)
index = VectorStoreIndex([docs])

#query_engine = index.as_query_engine()
retriver = VectorIndexRetriever(index=index,similarity_top_k=5)
query_engine = RetrieverQueryEngine(retriever=retriver)




def general_chat(prompt):
    messages = chat_prompt.format_messages(text=prompt)
    result = chat_model.invoke(messages)
    print("\n"+result.content+"\n")


def rag_chat(prompt):
    response = query_engine.query(prompt)
    print("\n",response)


def is_rag_query(prompt):
    rag_keywords = [
        "based on the document", "from the file", "according to the report", "as per the pdf",
        "from uploaded", "in the document", "mentioned in the file"
    ]
    return any(word in prompt.lower() for word in rag_keywords)

while True:
    prompt=input("Prompt: ")
    if prompt.lower()=="exit":
        break

    if is_rag_query(prompt):
        rag_chat(prompt)
    else:
        general_chat(prompt)