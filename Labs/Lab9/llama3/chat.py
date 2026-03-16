from ollama import chat

print("Chatbot ready. Type 'quit' to exit.")

history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    history.append({"role": "user", "content": user_input})

    response = chat(
        model="llama3.2:3b",
        messages=history,
    )

    reply = response.message.content
    history.append({"role": "assistant", "content": reply})

    print(f"\nBot: {reply}\n")