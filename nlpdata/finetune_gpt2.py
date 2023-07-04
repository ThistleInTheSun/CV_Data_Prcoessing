import os

import evaluate
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (GPT2ForSequenceClassification, GPT2Tokenizer,
                          TrainingArguments, get_scheduler)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

batch = 32
device_ids = [x for x in range(4)]
local_rank = "cuda:" + ",".join([str(x) for x in range(4)])

dataset = load_dataset("sst2")
# dataset = load_dataset("yelp_review_full")
# print(dataset)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
print(tokenizer)

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
# print("after map:", tokenized_datasets)
tokenized_datasets = tokenized_datasets.remove_columns(["idx"])
tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
# print("after rm:", tokenized_datasets)
for data in tokenized_datasets["validation"][:5]:
    print(data)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=16)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=16)



# model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=2)
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
for name, param in model.named_parameters():
    if name.startswith("transformer"):
        param.requires_grad = False
    # print(name, param.requires_grad)

model.config.pad_token_id = model.config.eos_token_id
# print("gpu:", print(local_rank))
# # torch.cuda.set_device(local_rank)
# # torch.distributed.init_process_group(backend='nccl')
# model = nn.DataParallel(model, device_ids)
model = model.cuda()  # 在使用DistributedDataParallel之前，需要先将模型放到GPU上
# model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print(model)


training_args = TrainingArguments(output_dir="test_trainer")
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

metric= load_metric("accuracy")
# progress_bar = tqdm(range(num_training_steps), desc="train")
best_acc = 0
val_each_step = 20
step = 0
for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_dataloader, desc="train"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        # progress_bar.update(1)
        step += 1

        if step % val_each_step != 0:
            continue
        model.eval()
        for batch in tqdm(eval_dataloader, desc="val:"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        res = metric.compute()
        print("epoch:", epoch, "loss:", loss, "acc:", res["accuracy"])
        if res["accuracy"] > best_acc:
            torch.save(model, 'best.pt')
torch.save(model, 'last.pt')



# model.eval()
# for batch in eval_dataloader:
#     batch = {k: v.to(device) for k, v in batch.items()}
#     with torch.no_grad():
#         outputs = model(**batch)

#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)
#     metric.add_batch(predictions=predictions, references=batch["labels"])

# metric.compute()


