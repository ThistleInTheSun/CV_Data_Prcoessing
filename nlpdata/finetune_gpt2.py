import os
import time

import evaluate
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch import nn
from torch.autograd import Variable
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (GPT2ForSequenceClassification, GPT2Tokenizer,
                          TrainingArguments, get_scheduler)

model_name = "gpt2-medium"
data_name = "sst2"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

batch_size = 8
save_project = "./work/{}-{}".format(model_name, data_name)
val_each_step = 500
num_epochs = 200
resume = True


def get_data():
    dataset = load_dataset(data_name)

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

    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=batch_size)
    return train_dataloader, eval_dataloader


def get_model():
    if resume:
        print("resume from:", resume)
        model = GPT2ForSequenceClassification.from_pretrained(save_project + "_last")
    else:
        model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=2)

    for name, param in model.named_parameters():
        if name.startswith("transformer"):
            param.requires_grad = False

    model.config.pad_token_id = model.config.eos_token_id
    if torch.cuda.is_available():
        model = model.cuda()
    if torch.cuda.device_count() > 1:
        print("use device:", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    return model


model = get_model()
train_dataloader, eval_dataloader = get_data()
training_args = TrainingArguments(output_dir="test_trainer")
optimizer = AdamW(model.parameters(), lr=5e-5)

num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

metric= load_metric("accuracy")
best_acc = 0
step = 0
save_list = ["epoch loss acc time"]
txt_path = os.path.join(save_project, "loss.txt")
with open(txt_path, "a") as f:
    if resume:
        f.write("\nresume from {}\n".format(resume))
    f.write("epoch loss acc time\n")
os.makedirs(save_project, exist_ok=True)
last_time = time.time()

for epoch in range(num_epochs):
    # train
    model.train()
    for batch in tqdm(train_dataloader, desc="train"):
        batch = {k: Variable(v.cuda()) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.mean().backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        step += 1

        # val
        if step % val_each_step != 0:
            continue
        model.eval()
        for batch in tqdm(eval_dataloader, desc="val:"):
            batch = {k: Variable(v.cuda()) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        res = metric.compute()

        # save
        time_each_step = (time.time() - last_time) / val_each_step
        last_time = time.time()
        print("epoch:", epoch, "loss:", loss.mean().item(), "acc:", res["accuracy"], 
              "time each step:", time_each_step, "\n")
        save_list.append([epoch, loss.mean().item(), res["accuracy"], time_each_step])
        if res["accuracy"] > best_acc:
            model.module.save_pretrained(save_project + "_best")
        with open(txt_path, "a") as f:
            line  = save_list[-1]
            line = " ".join([str(x) for x in line])
            if not line.endswith("\n"):
                line = line + "\n"
            f.write(line)
        model.module.save_pretrained(save_project + "_last")
