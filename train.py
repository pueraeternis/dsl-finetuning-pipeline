# isort: off
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
# isort: on

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer
from transformers.trainer_callback import EarlyStoppingCallback
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from src.logger import get_logger

logger = get_logger(__name__)

# --- Configuration ---
DATASET_PATH: Path = Path("data/dataset_train.json")
MAX_SEQ_LENGTH: int = 2048
NUM_TRAIN_EPOCHS: int = 3
BASE_MODEL_PATH: str = "unsloth/phi-4-unsloth-bnb-4bit"
OUTPUT_DIR: str = f"models/phi-4-auradsl-{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M')}"
FIXED_SEED: int = 42


def formatting_prompts_func(examples: dict[str, list[Any]], tokenizer: PreTrainedTokenizer):
    """Formats the dataset entries into the Phi-4 chat structure."""
    texts = []
    for context, input_text, output in zip(examples["context"], examples["input"], examples["output"], strict=True):
        messages = [
            {
                "role": "system",
                "content": "You are an expert in AuraDSL. Translate the natural language request into a valid AuraDSL query based on the provided schema.",
            },
            {"role": "user", "content": f"Context: {context}\n\nInput: {input_text}"},
            {"role": "assistant", "content": output},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}


def train() -> None:
    """Train Phi-4 to speak AuraDSL using Unsloth and LoRA."""
    # Load Model & Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,  # Auto-detection (BF16 on A100)
    )

    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "embed_tokens",
            "lm_head",
        ],
        use_rslora=True,
        bias="none",
        random_state=FIXED_SEED,
    )

    # Setup Chat Template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="phi-4",
    )

    # Data Preparation
    logger.info("Loading dataset from %s", DATASET_PATH)
    raw_dataset = Dataset.from_json(str(DATASET_PATH))

    # Split dataset to train/eval
    dataset = raw_dataset.train_test_split(test_size=0.1, seed=FIXED_SEED)  # pyright: ignore[reportAttributeAccessIssue]

    train_data = dataset["train"].map(
        lambda x: formatting_prompts_func(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    eval_data = dataset["test"].map(
        lambda x: formatting_prompts_func(x, tokenizer),
        batched=True,
        remove_columns=dataset["test"].column_names,
    )

    logger.info("Example prompt:\n%s", train_data[0]["text"])

    # Training Configuration
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        warmup_ratio=0.1,
        learning_rate=1e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        report_to="tensorboard",
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        seed=FIXED_SEED,
    )

    # Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=sft_config,
        processing_class=tokenizer,
    )

    # Add Early Stopping
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

    # Start Training
    logger.info("Starting training...")
    trainer.train()

    # Save Model
    logger.info("Saving model to %s", OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    train()
