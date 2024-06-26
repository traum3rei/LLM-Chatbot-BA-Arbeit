#!/usr/bin/env python
from typing import List
import asyncio
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langserve import add_routes
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
import chromadb
from chromadb.config import Settings


#DB
client = chromadb.HttpClient(settings=Settings())
collection =client.create_collection("docs")
collection.add(ids=[])

chat_history = []
from langchain.agents import tool
import chainlit as cl

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(data)
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = Chroma.from_documents(
    documents=documents,
    collection_name="rag-chroma",
    embedding=embeddings
)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

'''prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", MessagesPlaceholder(variable_name="input")),
])'''

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. Your name is Carl. Answer the questions with the help of the following context: {context}"),
    ("human", "{user_input}"),
])

llm = ChatOllama(model="llama3")

MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Du bist ein hilfreicher Mitarbeiter eines IT-Support Teams. Du hilfst der Firma intern bei IT-Problemen und beantwortest Fragen auf der Sprache auf der sie dir gestellt wurden.",
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


chain = (
    {"context": retriever, "user_input": RunnablePassthrough()}
    | prompt 
    | llm
    | StrOutputParser()
)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: x["intermediate_steps"],
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | StrOutputParser()
)

#agent_executor = AgentExecutor(agent=agent, verbose=True)

'''while True:
        inp = input()
        chain_output = chain.invoke(inp)
        print(chain_output)'''

'''async def main():
    print(chain.invoke("How are you doing?"))
    while True:
        inp = input()
        #chain_output = chain.invoke(inp)
        #print(chain_output)
        async for chunk in chain.astream(inp):
            print(chunk, end='')'''

@cl.step
def tool(message):
    print(message)
    return chain.invoke(message)

@cl.on_message
async def main(message: cl.Message):
    ans = tool(message.content)
    print(ans)
    await cl.Message(content=ans).send()


'''loop = asyncio.get_event_loop()
try:
    loop.run_until_complete(main())
except KeyboardInterrupt:
    pass
finally:
    loop.close()'''

'''# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route

# We need to add these input/output schemas because the current AgentExecutor
# is lacking in schemas.

class Input(BaseModel):
    input: str
    chat_history: List[BaseMessage] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "location"}},
    )


class Output(BaseModel):
    output: str

add_routes(
    app,
    chain,
    path="/agent",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)'''
