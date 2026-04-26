from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig

def format_example(example):
    return {
        "text": f"""### Instruction:
Solve the math problem step by step.

### Question:
{example['question']}

### Answer:
{example['answer']}"""
    }

def tokenize(example, tokenizer):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=256,
        padding=True
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

def prepare_model(model):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return get_peft_model(model, lora_config)

def get_trainer(model, dataset):
    training_args = TrainingArguments(
        output_dir="./qwen-gsm8k-lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
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

    dataset = dataset.map(format_example)
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