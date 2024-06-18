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
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

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
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
llm = ChatOllama(model="llama3")

MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            contextualize_q_system_prompt,
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
    ]
)

sys_prompt = "Du bist ein hilfreicher Mitarbeiter eines IT-Support Teams. Du hilfst der Firma intern bei IT-Problemen und beantwortest Fragen auf der Sprache auf der sie dir gestellt wurden. {context}"
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", sys_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
context_chain = template | llm | StrOutputParser()

def contextualized_question(input: dict):
    if input.get("chat_history"):
        return context_chain
    else:
        return input["question"]

chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | prompt 
    | llm
    | StrOutputParser()
)
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, prompt
)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

'''rag_chain = (
    RunnablePassthrough.assign(
        context=lambda input: contextualized_question(input) | retriever
    )
    | sys_prompt 
    | llm
)'''

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

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

@cl.step
def tool(message):
    print(message)
    return conversational_rag_chain.invoke(message)

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
