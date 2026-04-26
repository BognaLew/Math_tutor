from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig

from utils.inference import build_prompt


def tokenize(examples, tokenizer):
    prompts = [build_prompt(q) for q in examples['question']]
    full_texts = [prompt + answer 
                              for prompt, answer in 
                              zip(prompts, examples['answer'])]

    tokens = tokenizer(
        full_texts,
        truncation=True,
        max_length=256,
        padding='max_length'
    )

    prompt_tokens = tokenizer(
        prompts,
        truncation=True,
        max_length=256,
    )

    labels = []
    for i in range(len(full_texts)):
        input_ids = tokens["input_ids"][i]
        label = input_ids.copy()

        prompt_len = sum(
            1 for t in prompt_tokens["input_ids"][i]
            if t != tokenizer.pad_token_id
        )

        label[:prompt_len] = [-100] * prompt_len
        labels.append(label)

    tokens["labels"] = labels
    return tokens

def prepare_model(model):
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj"
            ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)

def get_trainer(model, dataset):
    training_args = TrainingArguments(
        output_dir="./qwen-gsm8k-lora", 
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4, 
        learning_rate=2e-4, 
        num_train_epochs=3, 
        warmup_steps=.05,
        fp16=True, 
        logging_steps=10, 
        save_steps=500, 
        optim="paged_adamw_8bit"
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

def fine_tune(model, tokenizer, dataset):
    model = prepare_model(model)

    tokenized = dataset.map(
        tokenize,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer}
    )

    trainer = get_trainer(model, tokenized)
    trainer.train()

    model.save_pretrained("math-tutor")
    tokenizer.save_pretrained("math-tutor")

    trained_model = model.merge_and_unload()
    trained_model.eval()
    return trained_model, tokenizer