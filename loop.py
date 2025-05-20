import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name_1B = 'meta-llama/Llama-3.2-1B-Instruct'
model_name_8B = 'meta-llama/Llama-3.2-3B-Instruct'

# Load model and tokenizer (in half precision)
model = AutoModelForCausalLM.from_pretrained(
    model_name_1B,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_name_1B)


# fonction to create a chat message
def format_prompt(message, history=[], system="You are a helpful assistant."):
    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"
    for user, assistant in history:
        prompt += f"{user.strip()} [/INST] {assistant.strip()} </s><s>[INST] "
    prompt += f"{message.strip()} [/INST]"
    return prompt

history = []

print("🦙 LLaMA-3 Chat (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    prompt = format_prompt(user_input, history)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    reply = decoded.split("[/INST]")[-1].split("</s>")[0].strip()
    history.append((user_input, reply))
    print(f"LLaMA: {reply}\n")
