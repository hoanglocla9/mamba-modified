
# from mamba_ssm.models.config_mamba import MambaConfig
# from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import torch
from transformers import GPTNeoXTokenizerFast, Trainer, TrainingArguments, Mamba2ForCausalLM, AutoConfig
from datasets import load_dataset

from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

def get_dataset(dataset_path="/root/llm_dataset/pile-uncopyrighted/", split="train"):
    if split == "train":
        dataset_path = dataset_path + "train/*.jsonl.zst"
    else:
        dataset_path = dataset_path + f"{split}.jsonl.zst"
    dataset = load_dataset('json', data_files={split:dataset_path}, split=split, streaming=True)
    return dataset

def get_tokenizer():
    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    return tokenizer

config = AutoConfig.from_pretrained('state-spaces/mamba2-130m')
model = Mamba2ForCausalLM(config)
model.to('cuda')
train_dataset = get_dataset(split="train")
valid_dataset = get_dataset(split="val")

output_dir = './mamba_ckpt'
tokenizer = get_tokenizer()
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" #enforce padding side left


# def tokenize_function(example):
#     """Tokenizes the dataset examples."""
#     return tokenizer(example["text"], truncation=True, max_length=1024, padding="max_length")

# tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
# tokenized_valid = valid_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# print(tokenized_valid)

training_args = SFTConfig(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=128,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-3,
    max_steps=4800,
    max_seq_length=1024
)
lora_config =  LoraConfig(
        r=8,
        target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
        task_type="CAUSAL_LM",
        bias="none"
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    dataset_text_field="text"
)
trainer.train()