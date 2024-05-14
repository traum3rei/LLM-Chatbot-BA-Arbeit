from langchain_community.llms import Ollama

def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append((" ").join(buffer))
                buffer = []
        if len(buffer):
            paragraphs.append((" ").join(buffer))
        return paragraphs

def chat():
    llm = Ollama(model="llama3")

    print(llm.invoke("Tell me a joke"))

def main():
    filename = "peterpan.txt"
    paragraphs = parse_file(filename)
    print(paragraphs[:10])

main()