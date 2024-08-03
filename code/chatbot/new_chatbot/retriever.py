from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings

class RetrieverManager:
    def __init__(self):



        client = chromadb.HttpClient(host="localhost", port=8000, settings=Settings(persist_directory="db"))

        emb_model = "intfloat/multilingual-e5-large-instruct"
        model_kwargs = {"device": "mps"}
        encode_kwargs = {"normalize_embeddings": True}

        embeddings = HuggingFaceEmbeddings(
                model_name=emb_model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
        )

        db = Chroma(embedding_function=embeddings, client=client, collection_name="docs")
        self.retriever = db.as_retriever()

    def get_retriever(self):
        return self.retriever


rvm = RetrieverManager()
ret = rvm.get_retriever()
