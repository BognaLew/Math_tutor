import re


def extract_answer(text):
    patterns = [
        r"answer is[:\s]*([-+]?\d+\.?\d*)",
        r"####\s*([-+]?\d+\.?\d*)"
    ]

    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return match.group(1)

    numbers = re.findall(r"[-+]?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]

    return None

def build_prompt(question):
    return f"""
        # Instruction:
        Solve step by step. Show all reasoning steps before giving the
        final answer.
        Final answer MUST be in format: #### number

        ## Question:
        {question}

        ## Answer:
        """

def generate_answer(question, model, tokenizer):
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=.7
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded