import json
import os

from .ollama_client import generate

from app.config import settings

DEFAULT_DATA_PATH = settings.data_path
DEFAULT_DATA_NAME = settings.history_file
MAX_MESSAGES = settings.max_messages

class ChatMemory:
    def __init__(self):
        print(os.getcwd())

        filepath = os.path.join(DEFAULT_DATA_PATH, DEFAULT_DATA_NAME)
        if os.path.isfile(filepath) and os.access(filepath, os.R_OK):
            try:
                with open(filepath, 'r') as file:
                    self.history = json.load(file)
            except:
                print("Failed to read file!")

        else:
            try:
                os.makedirs(DEFAULT_DATA_PATH, exist_ok=True)
            except:
                print("Directory Exists!")
            #For Chatbot Consistency
            self.history = [
                {
                    "role": "system",
                    "content": "You are a helpful personal assistant for a software developer"
                }
            ]

    def add_user(self, text):
        self.history.append({"role": "user", "content": text})

    def add_assistant(self, text):
        self.history.append({"role": "assistant", "content": text})

    def build_prompt(self):
        return "\n".join(
            f"{m['role']}: {m['content']}" for m in self.history
        )
    
    def summarize_memory(self, messages, model):
        conversation = "\n".join(
            [f"{m['role']}: {m['content']}" for m in messages]
        )

        prompt = f"""
        Summarize the following conversation briefly while preserving important facts.

        Conversation:
        {conversation}

        Summary:
        """

        summary = generate(model,prompt)

        return summary
    
    def maybe_compress(self, model):
        if len(self.history) > MAX_MESSAGES:

            old_messages = self.history[:-4]
            recent_messages = self.history[-4:]

            summary = self.summarize_memory(old_messages, model)

            self.history = [
                {
                    "role": "system",
                    "content": f"Conversation summary: {summary}"
                }
            ] + recent_messages
    
    def save_history(self):
        filepath = DEFAULT_DATA_PATH + DEFAULT_DATA_NAME
        with open(filepath, 'w') as file:
            converted_json = json.dumps(self.history)
            file.write(converted_json)