import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, Trainer, TrainingArguments


# read dataset from the huggingface hub
dataset = load_dataset("NickyNicky/nlp-mental-health-conversations", split="train")

# load the tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token 

def tokenize_function(examples):
    tokens = tokenizer(examples["Context"], truncation=True, padding="max_length", max_length=128)
    tokens['labels'] = tokens['input_ids'].copy()  # label are the same as the input IDs 
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# load the model from pretrained GPT-2 in huggingface
model = GPT2LMHeadModel.from_pretrained("distilgpt2")

# set device (GPU or CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

# finetune the model
training_args = TrainingArguments(
    output_dir="./health_gpt",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    prediction_loss_only=True,
    fp16=True  
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()

# Generate text
prompt = "The reason for my depression may be"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=50, temperature=0.7)
print(tokenizer.decode(outputs[0]))