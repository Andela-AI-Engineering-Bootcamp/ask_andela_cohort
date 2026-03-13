# Login: opens a prompt for your HF token (get one at https://huggingface.co/settings/tokens)
from huggingface_hub import login
# Upload dataset to HF and return repo id
import json
from pathlib import Path
from datasets import Dataset, load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training 

login()  # paste token when prompted; use token with "write" scope to upload

# Constants
HF_USERNAME = "Karosi"  # change to your HF username
REPO_NAME = "ask-andela-finetune-dataset"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
USE_HF_DATASET = True  # False = use DATASET_PATH below
REPO_ID = "Karosi/ask-andela-finetune-dataset"  # from upload cell output
OUTPUT_DIR = "checkpoints/ask_andela_lora"
USE_4BIT = True
MAX_SEQ_LEN = 512
EPOCHS = 3
BATCH = 2
GRAD_ACCUM = 4
LR = 2e-5

# Path to your JSONL (in Colab: upload the file or clone the repo first)
DATASET_PATH = "data/finetune_dataset.jsonl"

rows = []
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

ds = Dataset.from_list(rows)
ds.push_to_hub(REPO_ID, private=False)
print(f"Dataset pushed to https://huggingface.co/datasets/{REPO_ID}")
print(f"Repo ID: {REPO_ID}")

def load_jsonl_messages(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            if "messages" in obj: examples.append(obj["messages"])
    return examples

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

model_kwargs = {"trust_remote_code": True}
if USE_4BIT:
    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
if USE_4BIT: model = prepare_model_for_kbit_training(model)

model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], bias="none", task_type="CAUSAL_LM"))
model.print_trainable_parameters()


if USE_HF_DATASET:
    hf_ds = load_dataset(REPO_ID, split="train")
    messages_list = [ex["messages"] for ex in hf_ds]
else:
    messages_list = load_jsonl_messages(DATASET_PATH)

texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in messages_list]
print(f"Examples: {len(texts)}")

def tokenize(examples):
    out = tokenizer(examples["text"], truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")
    pad_id = tokenizer.pad_token_id
    out["labels"] = [[tid if tid != pad_id else -100 for tid in ids] for ids in out["input_ids"]]
    return out

ds = Dataset.from_dict({"text": texts})
tokenized = ds.map(lambda x: tokenize({"text": x["text"]}), batched=True, remove_columns=ds.column_names)
tokenized.set_format("torch")

# Cell 6: Train + save
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH, gradient_accumulation_steps=GRAD_ACCUM, learning_rate=LR, bf16=True, logging_steps=5, save_steps=50, save_total_limit=2, report_to="none"),
    train_dataset=tokenized,
)
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}")
trainer.push_to_hub()
model_repo_name = Path(OUTPUT_DIR).name
print(f"Model uploaded to: https://huggingface.co/{HF_USERNAME}/{model_repo_name}")