# AI-Powered-Chatbot-
This chatbot combines general knowledge answering with document-specific querying capabilities. Users can upload documents, and the bot will retrieve relevant information from those files to generate precise, context-aware responses.
This project is a versatile chatbot designed to handle general conversational queries as well as specific questions based on user-uploaded documents. It seamlessly combines powerful language models with retrieval-augmented generation (RAG) techniques to deliver accurate, context-aware responses.

Features
General Question Answering:
Utilizes LangChain framework powered by the Gemini 1.5 Flash LLM API to generate natural, conversational responses to a wide range of general questions.

Document-Based Querying (RAG):
Employs Llama Index (formerly GPT Index) to process and index user-uploaded documents, enabling the chatbot to answer queries grounded in the uploaded content with precision.

Single Unified Interface:
Routes user prompts intelligently to either the general LLM pipeline or the document-based RAG pipeline based on the query context.

Powered by Gemini 1.5 Flash API:
Both general and RAG responses leverage the Gemini 1.5 Flash API for fast and high-quality language understanding and generation.

Technologies Used
LangChain — for managing general language model interactions

Llama Index — for document indexing and retrieval-augmented generation

Gemini 1.5 Flash API — the underlying large language model API powering both pipelines

Python — main programming language

[Additional dependencies / frameworks you used can be listed here]

How to Use
Clone the repository

Install dependencies (pip install -r requirements.txt)

Set your Gemini 1.5 Flash API key as an environment variable

Run the chatbot application

Upload documents as needed and start querying!
