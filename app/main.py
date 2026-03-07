from chatbot import ChatBot

bot = ChatBot()

print("Personal AI ready. Type 'exit' to quit.")

while True:
    user = input("You: ")

    if user == "exit":
        bot.save()
        break

    reply = bot.chat(user)

    print("Bot:", reply)