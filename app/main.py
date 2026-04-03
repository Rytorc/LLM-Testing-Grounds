from .chatbot import ChatBot

bot = ChatBot()

print("Personal AI ready. Type 'exit' to quit.")

while True:
    user = input("You: ").strip()

    if user.lower() == "exit":
        bot.save()
        break

    reply = bot.chat(user)
    print(f"Bot: {reply}")