from langchain_community.llms import Ollama

class LLMManager:
    def __init__(self):
        self.llm = Ollama(model="llama3")

    def get_llm(self):
        return self.llm