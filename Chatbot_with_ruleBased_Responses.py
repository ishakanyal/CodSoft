def ruleBased_chatbot(user_input):
    # Convert user input to lowercase for easier comparison
    user_input = user_input.lower()

    # Define the predefined rules and responses
    if "hello" in user_input or "hi" in user_input:
        response = "Hello there! How can I help you?"
    elif "how are you?" in user_input :
        response = "I'm just a chatbot, but I'm here to assist you!"
    elif "what's your name" in user_input or "who are you" in user_input:
        response = "I'm a simple chatbot. You can call me ChatBot."
    elif "what is my name" in user_input or "do you know me?" in user_input:
        response = "Your name is Isha Kanyal. You are from Uttarakhand."
    elif "ok bye chatbot" in user_input or "bye" in user_input:
        response = "Goodbye! Have a great day!"
    else:
        response = "I'm sorry, can you explain it. I don't understand that :("

    return response

# Main loop for chatting
print("ChatBot: Hi! Type 'Hello' to start the chat or type 'Bye' to end the chat.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "bye":
        print("ChatBot: Goodbye!! Have a nice day ;)")
        break
    else:
        response = ruleBased_chatbot(user_input)
        print("ChatBot:", response)
