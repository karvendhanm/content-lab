from litellm import completion

email_text = (
    "from: johndoe@gmail.com"
    "Hi Jane Doe,"
    "Request you to send me a shipment of snake oil as early as possible.\n\n"
    "Regards,"
    "John Doe"
)


response = completion(
    model="ollama/qwen2.5-coder:14b",
    api_base="http://localhost:11434",
    messages=[
        {"role": "system", "content": "You are an assistant that summarizes emails."},
        {
            "role": "user",
            "content": "Extract the sender's name and summarize the key results"
            + email_text,
        },
    ],
    temperature=0,
)


print(response.choices[0].message.content)
# Example output
# **Sender's Name:** John Doe

# **Summary of Key Results:**
# John Doe has requested Jane Doe to send a shipment of snake oil as soon as possible.
