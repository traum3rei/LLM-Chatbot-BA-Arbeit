import os
import langchain as cl
from langchain import LLMChain, RetrievalChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import ChromaDB
from langchain.document_loaders import DirectoryLoader
from ollama import Ollama

# Load the LLaMA 3 model locally using Ollama
model_path = 'path/to/llama3-model'
ollama = Ollama("llama3")

# Define a template for the prompt
prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="""
    Du bist ein IT-Support-Mitarbeiter. Beantworte die folgende Frage basierend auf den bereitgestellten Informationen.

    Frage: {question}
    Kontext: {context}
    
    Antwort auf Deutsch:
    """
)

# Initialize the ChromaDB client and load company data
chroma_client = ChromaDB()
loader = DirectoryLoader('path/to/company/data')
documents = loader.load()
chroma_client.index(documents)

# Define the Retrieval-Augmented Generation (RAG) chain
class LocalOllama:
    def __init__(self, ollama):
        self.ollama = ollama

    def generate(self, prompt):
        response = self.ollama.generate(prompt, max_length=500)
        return response['text']

local_ollama = LocalOllama(ollama)

rag_chain = RetrievalChain(
    llm_chain=LLMChain(
        llm=local_ollama,
        prompt=prompt_template
    ),
    retriever=chroma_client.as_retriever()
)

# Define the Chainlit app
app = Chainlit()

@cl.step
def tool(message):
    print(message)
    return agent.invoke(message)

@cl.on_message
async def main(message: cl.Message):
    ans = tool(message.content)
    print(ans)
    await cl.Message(content=ans).send()

@app.route('/')
def chatbot(query: str):
    response = rag_chain.run(question=query)
    return response

# Run the app
if __name__ == '__main__':
    app.run(port=8000)

