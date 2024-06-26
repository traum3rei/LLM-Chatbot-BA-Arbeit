import chromadb
from chromadb.config import Settings


#DB
client = chromadb.HttpClient(settings=Settings())
collection =client.create_collection("docs")
collection.add(ids=[1, 2], documents=["a", "b"])
docs = []

db = Chroma.from_documents(
    docs,
    embeddings,
    client=client,
    collection_name="docs",
    
)