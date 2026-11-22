from litellm import completion

email_text = (
    "from: johndoe@gmail.com"
    "Hi Jane Doe,"
    "Request you to send me a shipment of snake oil as early as possible.\n\n"
    "If you would like to get in touch with me, then call me on my mobile: 90075 00759 \n\n"
    "Regards,"
    "John Doe"
)


messages = [
    {
        "role": "system",
        "content": "You are an assistant that summarizes and answers questions about "
        "emails.",
    },
    {
        "role": "user",
        "content": "Extract the sender's name and summarize the key results, and "
        "here is the email content: " + email_text,
    },
]

response1 = completion(
    model="ollama/qwen2.5-coder:14b",
    api_base="http://localhost:11434",
    messages=messages,
)

messages.append({"role": "assistant", "content": response1.choices[0].message.content})

messages.append(
    {
        "role": "user",
        "content": "Is there any mention of preferred contact method ?",
    },
)

response2 = completion(
    model="ollama/qwen2.5-coder:14b",
    api_base="http://localhost:11434",
    messages=messages,
)

print(f"the model response is: {response2.choices[0].message.content}")
