import os
import json
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    set_seed
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import DataCollatorForLanguageModeling
from functools import partial

# Set environment variables
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data directory and files
data_dir = "Data"
json_files = ["Data/FAQ.json"]

# Load JSON file
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        faqs = data['qa_pairs']
    return faqs

# Load data
all_data = []
for file in json_files:
    fqas = load_json_data(file)
    for fqa in fqas:
        all_data.append(fqa)

# Convert to datasets format
dataset = Dataset.from_pandas(pd.DataFrame(all_data))

# Print dataset information
print("Dataset:", dataset)
print("Number of samples in the dataset:", len(dataset))
print("First few samples in the dataset:")
for i in range(min(5, len(dataset))):
    print(dataset[i])

# Split the dataset
if len(dataset) > 1:
    train_testvalid_split = dataset.train_test_split(test_size=0.2)
    dataset_dict = DatasetDict({
        'train': train_testvalid_split['train'],
        'valid': train_testvalid_split['test']
    })
else:
    raise ValueError("Not enough samples to split into train and validation sets.")

# Model path
model_name = "autodl-tmp/llama2/meta-llama_Llama-2-7b"

# Create directory
os.makedirs(model_name, exist_ok=True)

# Load and save Tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b", trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.save_pretrained(model_name)

# Load and save model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
model.save_pretrained(model_name)

# Configure BitsAndBytes
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False
)

# Load pretrained model
original_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_cache=False,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map={"": 0}
)

# Define generation function
eval_tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=True, trust_remote_code=True, use_fast=False)
eval_tokenizer.pad_token = eval_tokenizer.eos_token

def gen(model, p, maxlen=100, sample=True):
    toks = eval_tokenizer(p, return_tensors="pt").to(device)
    res = model.generate(**toks, max_new_tokens=maxlen, do_sample=sample,
                         num_return_sequences=1, temperature=0.1, num_beams=1, top_p=0.95)
    return eval_tokenizer.batch_decode(res, skip_special_tokens=True)

# Set random seed
seed = 42
set_seed(seed)

# Test data
index = 9
prompt = dataset_dict['valid'][index]['question']
answer = dataset_dict['valid'][index]['answer']
formatted_prompt = f"Instruct: Answer the questions.\n{prompt}\nOutput:\n"
res = gen(original_model, formatted_prompt, 100)
output = res[0].split('Output:\n')[1]

# Print input and output
dash_line = '-' * 100
print(dash_line)
print(f'INPUT PROMPT:\n{formatted_prompt}')
print(dash_line)
print(f'BASELINE HUMAN :\n{answer}\n')
print(dash_line)
print(f'MODEL GENERATION - ZERO SHOT:\n{output}')

# Create prompt format function
def create_prompt_formats(sample):
    INTRO_BLURB = "Below is an instruction that describes a question. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruct: Answer the question."
    RESPONSE_KEY = "### Output:"
    END_KEY = "### End"
    
    blurb = f"\n{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}"
    input_context = f"{sample['question']}" if sample else None
    response = f"{RESPONSE_KEY}\n{sample['answer']}\n{END_KEY}"
    
    parts = [part for part in [blurb, instruction, input_context, response] if part]
    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt

    return sample

# Get model max length
def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break
    if not max_length:
        max_length = 50
        print(f"Using default max length: {max_length}")
    return max_length

# Preprocess batch data
def preprocess_batch(batch, tokenizer, max_length):
    return tokenizer(batch["text"], max_length=max_length, truncation=True, padding="max_length")

# Preprocess dataset
def preprocess_dataset(tokenizer, max_length, seed, dataset):
    print(f"Original dataset sample count: {len(dataset)}")
    dataset = dataset.map(create_prompt_formats)
    print(f"Sample count after creating prompt formats: {len(dataset)}")
    
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(_preprocessing_function, batched=True, remove_columns=['question', 'answer'])
    print(f"Sample count after tokenization: {len(dataset)}")
    
    # Print input_ids length for each sample
    for sample in dataset:
        print(f"Sample input_ids length: {len(sample['input_ids'])}")
    
    # Filter samples with input_ids length exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) <= max_length)
    print(f"Sample count after filtering: {len(dataset)}")
    
    # Shuffle the dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset

max_length = 1024
train_dataset = preprocess_dataset(tokenizer, max_length, seed, dataset_dict['train'])
eval_dataset = preprocess_dataset(tokenizer, max_length, seed, dataset_dict['valid'])

# Print the sample count of the datasets to confirm no data loss
print(f"Training set sample count after preprocessing: {len(train_dataset)}")
print(f"Validation set sample count after preprocessing: {len(eval_dataset)}")

# Configure LoRA
config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'dense'],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

original_model.gradient_checkpointing_enable()
original_model = prepare_model_for_kbit_training(original_model)
peft_model = get_peft_model(original_model, config)

# Print the number of trainable parameters
def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"Trainable model parameters: {trainable_model_params}\nAll model parameters: {all_model_params}\nPercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(peft_model))

# Training arguments configuration
output_dir = 'models/checkpoints'
training_args = TrainingArguments(
    output_dir=output_dir,
    warmup_steps=0,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    max_steps=50,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    logging_steps=5,
    logging_dir="./logs",
    save_strategy="steps",
    save_steps=5,
    eval_strategy="steps",
    eval_steps=5,
    do_eval=True,
    gradient_checkpointing=True,
    report_to="none",
    overwrite_output_dir=True,
    group_by_length=True,
    fp16=True,
)

trainer = Trainer(
    model=peft_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# Save the trained model and tokenizer
trained_model_dir = 'models/trained_model'
os.makedirs(trained_model_dir, exist_ok=True)
peft_model.save_pretrained(trained_model_dir)
tokenizer.save_pretrained(trained_model_dir)

print(f"Model saved at {trained_model_dir}")
