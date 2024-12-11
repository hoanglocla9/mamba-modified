import torch
from transformers import GPTNeoXTokenizerFast, Trainer, TrainingArguments, Mamba2ForCausalLM, AutoConfig, AutoModelForCausalLM
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
import os 
from accelerate import PartialState
device_string = PartialState().process_index

#ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)

from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

def get_dataset(dataset_path="/home/lhloc249/D1/llm_dataset/pile-uncopyrighted/", split="train", shuffle=True):
    if split == "train":
        dataset_path = dataset_path + "train/*.jsonl.zst"
    else:
        dataset_path = dataset_path + f"{split}.jsonl.zst"
    dataset = load_dataset('json', data_files={split:dataset_path}, split=split, streaming=True)
    if shuffle:
        return dataset.shuffle(buffer_size=10000, seed=42)
    return dataset

def get_tokenizer():
    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    return tokenizer

config = AutoConfig.from_pretrained('state-spaces/mamba-130m-hf')
model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", use_cache=False, device_map={'':device_string})
print("Number of Parameters:", model.num_parameters())
model.to('cuda')
train_dataset = get_dataset(split="train", shuffle=True) 
valid_dataset = get_dataset(split="val", shuffle=True)

train_dataset = train_dataset.skip(1000)
valid_dataset = valid_dataset.take(1000)
print("Rank: ", int(os.environ["RANK"]), "WORLD_SIZE", int(os.environ["WORLD_SIZE"]))
train_dataset = split_dataset_by_node(train_dataset, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
valid_dataset = split_dataset_by_node(valid_dataset, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))

output_dir = './mamba_ckpt'
tokenizer = get_tokenizer()
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" #enforce padding side left
gradient_checkpointing_kwargs={'use_reentrant':False} 

training_args = SFTConfig(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=64,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=6e-4,
    max_steps=7000, # 4800
    max_seq_length=1024,
    fp16=True,
    #fp16=True,
    ddp_find_unused_parameters=True,
#    torch_compile=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",    # Evaluate during training
    eval_steps=500,                 # Perform evaluation every 500 steps
    save_strategy="steps",          # Save checkpoints during training
    save_steps=1000,
    save_total_limit=3,             # Keep only 3 checkpoints
    warmup_steps=500,
    report_to="tensorboard",        # Log training progress to TensorBoard
    dataloader_num_workers=4,
    packing=True
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
