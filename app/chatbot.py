from memory import ChatMemory
from ollama_client import generate
from rag import search

class ChatBot:

    def __init__(self, model="llama3.1"):
        self.memory = ChatMemory()
        self.model = model

    def chat(self, user_input):

        self.memory.add_user(user_input)

        context_docs = search(user_input)
        context = "\n".join(context_docs) if context_docs else ""

        memory_prompt = self.memory.build_prompt()

        prompt = f"""
        Context:
        {context}

        Conversation:
        {memory_prompt}
        """

        reply = generate(self.model, prompt)

        self.memory.add_assistant(reply)

        return reply
    
    def save(self):
        self.memory.save_history()