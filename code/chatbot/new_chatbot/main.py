from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from chatbot_prompt import PromptManager
from llm_manager import LLMManager
from retriever import RetrieverManager
from session_manager import SessionManager
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_session_history(session_id):
        return SQLChatMessageHistory(session_id, "sqlite:///memory.db")

def handleMessage():
    ses = SessionManager()
    llm = LLMManager()
    prompt = PromptManager()
    ret = RetrieverManager()
    qa_chain = create_stuff_documents_chain(llm.llm, prompt.prompt)
    rag_chain = create_retrieval_chain(ret.retriever, qa_chain)

    with_message_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


    ans = with_message_history.invoke({"input": "Help me"}, {
            "configurable": {"session_id": 1
        }})

    print(ans)
    return ans

handleMessage()