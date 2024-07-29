from fastapi import FastAPI
import chainlit as cl
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from ragas.run_config import RunConfig
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
import pandas as pd

import ssl
import os
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import chromadb
from chromadb.config import Settings

app = FastAPI()

client = chromadb.HttpClient(host="localhost", port=8080, settings=Settings(persist_directory="db"))

emb_model = "intfloat/multilingual-e5-large-instruct"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}

embeddings = HuggingFaceEmbeddings(
        model_name=emb_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
)

db = Chroma(embedding_function=embeddings, client=client, collection_name="docs")
retriever = db.as_retriever()

# Create chatbot prompt

prompt = ChatPromptTemplate.from_messages([("system", """Dein Name ist Arthur.<D-s>Du bist ein erfahrenes Mitglied des IT-Support-Teams bei TechDivision. Deine Aufgabe ist es, Mitarbeitern bei ihren IT-Problemen zu helfen. Beachte folgende Richtlinien: 1. Nutze ausschließlich die Informationen aus dem bereitgestellten Kontext und deinem vorhandenen Wissen über allgemeine IT-Konzepte, um Fragen zu beantworten. 2. Wenn die Antwort nicht im Kontext oder in deinem allgemeinen IT-Wissen enthalten ist, sage ehrlich, dass du die Information nicht hast oder die Frage nicht beantworten kannst. 3. Erfinde oder halluziniere keine Informationen. 4. Beziehe dich auf frühere Teile der Konversation, wenn sie relevant sind. 5. Bleibe stets höflich, professionell und hilfsbereit. 6. Wenn nötig, frage nach weiteren Details, um das Problem besser zu verstehen. 7. Gib schrittweise Anleitungen, wenn du eine Lösung vorschlägst. 8. Verweise den Mitarbeiter an andere Abteilungen oder Experten, wenn das Problem außerhalb deines Zuständigkeitsbereichs liegt. 9. Antworte auf Deutsch. Kontext: {context} Bisheriger Chatverlauf: {chat_history} Beantworte die folgende Frage des Mitarbeiters basierend auf diesen Informationen."""),
                                            MessagesPlaceholder(variable_name="chat_history"),("user", "Frage: {input}")
                                            ])

# Use a pipeline as a high-level helper
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch

llm = Ollama(model="llama3")
llm_two = Ollama(model="phi3:medium")
output_parser = StrOutputParser()

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def format_docs(docs):
    print(docs)
    return "\n\n".join(doc.page_content for doc in docs)


qa_chain = create_stuff_documents_chain(llm, prompt)
rag_2_chain = create_retrieval_chain(retriever, qa_chain)

qa_3_chain = create_stuff_documents_chain(llm_two, prompt)
rag_3_chain = create_retrieval_chain(retriever, qa_3_chain)


from langchain_core.runnables.history import RunnableWithMessageHistory
from fastapi import Request
from langchain_community.chat_message_histories import SQLChatMessageHistory


message_db = Chroma(embedding_function=embeddings, client=client, collection_name="docs")

def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")

chain = (rag_2_chain | output_parser)
with_message_history = RunnableWithMessageHistory(
    rag_2_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

with_message_history_2 = RunnableWithMessageHistory(
    rag_3_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
#answer = chain.invoke({"input": "Hello"})
#print(answer)

from apiclient.discovery import build
from google.oauth2 import service_account

SCOPES = ['https://www.googleapis.com/auth/chat.bot']

# Specify service account details.
CREDENTIALS = service_account.Credentials.from_service_account_file(
    'credentials.json', scopes=SCOPES)

#chat = build('chat', 'v1', credentials=CREDENTIALS)
chat = build('chat', 'v1', credentials=CREDENTIALS)
@app.post("/")
async def root(request: Request):
    req = await request.json()

    text= req["message"]["text"]
    user = req["message"]["sender"]

    thread_name = req["message"]["thread"]["name"]
    thread = req["message"]["thread"]["name"].split("/")[3]
    print(thread)
    email_host = user["email"].split("@")

    if(email_host[1] != "techdivision.de" and email_host[1] != "gsuite-test.tdintern.de"):
        return {"text": "wrong user"}

    s_id = hash(thread)



    if ("@Arthur" in text):
        text = text.replace("@Arthur", "")

    ans = ""
    if ("llama" in text):
        text = text.replace("llama", "")
        ans = with_message_history.invoke({"input": text}, {
            "configurable": {"session_id": s_id
        }})
    elif ("phi" in text):
        text = text.replace("phi", "")
        ans = with_message_history_2.invoke({"input": text}, {
            "configurable": {"session_id": s_id
        }})
    else:
        ans = with_message_history.invoke({"input": text}, {
            "configurable": {"session_id": s_id
        }})

    print(ans)
    # Create a Chat message.
    chat.spaces().messages().create(

        # The space to create the message in.
        #
        # Replace SPACE with a space name.
        # Obtain the space name from the spaces resource of Chat API,
        # or from a space's URL.
        parent=req["message"]["space"]["name"],
        messageReplyOption='REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD',
        # The message to create.
        body={"text": "<"+user["name"]+"> "+ans["answer"], "thread": {"name": thread_name}}

    ).execute()
    return json.jsonify({})