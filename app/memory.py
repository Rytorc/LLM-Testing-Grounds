import json
import os

DEFAULT_DATA_PATH = "data/"
DEFAULT_DATA_NAME = "history.json"

class ChatMemory:
    def __init__(self):
        print(os.getcwd())

        filepath = DEFAULT_DATA_PATH + DEFAULT_DATA_NAME
        if os.path.isfile(filepath) and os.access(filepath, os.R_OK):
            try:
                with open(filepath, 'r') as file:
                    self.history = json.load(file)
            except:
                print("Failed to read file!")

        else:
            try:
                os.mkdir(DEFAULT_DATA_PATH)
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
    
    def save_history(self):
        filepath = DEFAULT_DATA_PATH + DEFAULT_DATA_NAME
        with open(filepath, 'w') as file:
            converted_json = json.dumps(self.history)
            file.write(converted_json)