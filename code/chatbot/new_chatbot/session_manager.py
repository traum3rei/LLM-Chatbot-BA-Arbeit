from langchain_community.chat_message_histories import SQLChatMessageHistory

class SessionManager:
    def get_session_history(session_id):
        return SQLChatMessageHistory(session_id, "sqlite:///memory.db")