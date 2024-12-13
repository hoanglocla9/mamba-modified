import torch
from transformers import GPTNeoXTokenizerFast, Trainer, TrainingArguments, Mamba2ForCausalLM, AutoConfig, AutoModelForCausalLM, Mamba2Config
from transformers.models.mamba2 import Mamba2Config
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
import os 
from accelerate import PartialState
device_string = PartialState().process_index

#ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)

from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

def get_dataset(dataset_path="/root/llm_dataset/pile-uncopyrighted/", split="train", shuffle=True):
    if split == "train":
        dataset_path = dataset_path + "train/*.jsonl.zst"
    else:
        dataset_path = dataset_path + f"{split}.jsonl.zst"
    dataset = load_dataset('json', data_files={split:dataset_path}, split=split, streaming=True)
    if shuffle:
        return dataset.shuffle(buffer_size=10_000, seed=42)
    return dataset

def get_tokenizer():
    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    return tokenizer

##config = AutoConfig.from_pretrained('state-spaces/mamba-130m-hf')
config = Mamba2Config(
	num_heads=24, ## num_heads = hidden_size * expand / head_dim
        head_dim=64,
        vocab_size=50277,
        hidden_size=768,
        state_size=128,
        num_hidden_layers=8,
        layer_norm_epsilon=1e-5,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        expand=2,
        conv_kernel=4,
        n_groups=1,
        use_bias=False,
        use_conv_bias=True,
        hidden_act="silu",
        initializer_range=0.1,
        residual_in_fp32=True,
        time_step_rank="auto",
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_floor=1e-4,
        time_step_limit=(0.0, float("inf")),
        rescale_prenorm_residual=False,
        use_cache=False,
        rms_norm=True,
        chunk_size=256,
        tie_word_embeddings=True
) 
model = AutoModelForCausalLM.from_config(config)  #AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", use_cache=False, device_map={'':device_string})

model_name = "mamba2_" + str(int(model.num_parameters())/1000000) + "M"
print("Model Name", model_name)
print("Number of Parameters:", model.num_parameters())
model.to('cuda')
train_dataset = get_dataset(split="train", shuffle=False) 
valid_dataset = get_dataset(split="val", shuffle=False)
#train_dataset = train_dataset.shard(num_shards=16, index=0)
#valid_dataset = valid_dataset.shard(num_shards=16, index=0)
#train_dataset = train_dataset.shuffle(buffer_size=10_000, seed=42)
#valid_dataset = valid_dataset.shuffle(buffer_size=10_000, seed=42)
#train_dataset = train_dataset.skip(1000)
#valid_dataset = valid_dataset.take(1000)
#print("Rank: ", int(os.environ["RANK"]), "WORLD_SIZE", int(os.environ["WORLD_SIZE"]))
train_dataset = split_dataset_by_node(train_dataset, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
valid_dataset = split_dataset_by_node(valid_dataset, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))

output_dir = './mamba_ckpt'
tokenizer = get_tokenizer()
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" #enforce padding side left
gradient_checkpointing_kwargs={'use_reentrant':False} 
bs = 24
#num_gpus = 2
training_args = SFTConfig(
    output_dir=f"./results/{model_name}",
    num_train_epochs=3,
    per_device_train_batch_size=bs,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=6e-4,
    max_steps=int(4800*512/bs/int(os.environ["WORLD_SIZE"])), # 4800
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
    dataloader_num_workers=0,
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
