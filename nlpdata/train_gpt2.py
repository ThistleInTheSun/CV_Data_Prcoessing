import os
import time

import evaluate
import numpy as np
import torch
import transformers
from datasets import load_dataset, load_metric
from torch import nn
from torch.autograd import Variable
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (GPT2ForSequenceClassification, GPT2Tokenizer,
                          TrainingArguments, get_scheduler)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'


model_name = "gpt2"
data_name = "sst2"
batch_size = 8
save_project = "./work/{}-{}".format(model_name, data_name)
val_each_step = 500
num_epochs = 200
resume = True


def get_model():
    if resume:
        print("resume from:", resume)
        model = GPT2ForSequenceClassification.from_pretrained(save_project + "_last")
        print("load done")
    else:
        model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # for name, param in model.named_parameters():
    #     if name.startswith("transformer"):
    #         param.requires_grad = False

    model.config.pad_token_id = model.config.eos_token_id
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    if fp16:
        try:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
        except ImportError:
            print("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    if torch.cuda.device_count() > 1:
        print("use device:", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    return model


def get_data():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(tokenizer)

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True)

    dataset = load_dataset(data_name)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["idx"])
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    for data in tokenized_datasets["validation"][:5]:
        print(data)

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)

    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=batch_size)
    return train_dataloader, eval_dataloader


def get_parameters(model):
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of parameters: {}'.format(num_parameters))


model = get_model()
get_parameters()
train_dataloader, eval_dataloader = get_data()

optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps,
                                                        t_total=total_steps)

print('starting training')
overall_step = 0
running_loss = 0
for epoch in range(num_epochs):
    print('epoch {}'.format(epoch + 1))
    x = np.linspace(0, num_pieces - 1, num_pieces, dtype=np.int32)
    random.shuffle(x)
    piece_num = 0
    for i in x:
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
            line = f.read().strip()
        tokens = line.split()
        tokens = [int(token) for token in tokens]
        start_point = 0
        samples = []
        while start_point < len(tokens) - n_ctx:
            samples.append(tokens[start_point: start_point + n_ctx])
            start_point += stride
        if start_point < len(tokens):
            samples.append(tokens[len(tokens)-n_ctx:])
        random.shuffle(samples)
        for step in range(len(samples) // batch_size):  # drop last

            #  prepare data
            batch = samples[step * batch_size: (step + 1) * batch_size]
            batch_inputs = []
            for ids in batch:
                int_ids = [int(x) for x in ids]
                batch_inputs.append(int_ids)
            batch_inputs = torch.tensor(batch_inputs).long().to(device)

            #  forward pass
            outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)
            loss, logits = outputs[:2]

            #  get loss
            if multi_gpu:
                loss = loss.mean()
            if gradient_accumulation > 1:
                loss = loss / gradient_accumulation

            #  loss backward
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            #  optimizer step
            if (overall_step + 1) % gradient_accumulation == 0:
                running_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            if (overall_step + 1) % log_step == 0:
                tb_writer.add_scalar('loss', loss.item() * gradient_accumulation, overall_step)
                print('now time: {}:{}. Step {} of piece {} of epoch {}, loss {}'.format(
                    datetime.now().hour,
                    datetime.now().minute,
                    step + 1,
                    piece_num,
                    epoch + 1,
                    running_loss * gradient_accumulation / (log_step / gradient_accumulation)))
                running_loss = 0
            overall_step += 1
        piece_num += 1

    print('saving model for epoch {}'.format(epoch + 1))
    if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
        os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))
    # torch.save(scheduler.state_dict(), output_dir + 'model_epoch{}/scheduler.pt'.format(epoch + 1))
    # torch.save(optimizer.state_dict(), output_dir + 'model_epoch{}/optimizer.pt'.format(epoch + 1))
    print('epoch {} finished'.format(epoch + 1))

    then = datetime.now()
    print('time: {}'.format(then))
    print('time for one epoch: {}'.format(then - now))

print('training finished')
if not os.path.exists(output_dir + 'final_model'):
    os.mkdir(output_dir + 'final_model')
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(output_dir + 'final_model')
# torch.save(scheduler.state_dict(), output_dir + 'final_model/scheduler.pt')
# torch.save(optimizer.state_dict(), output_dir + 'final_model/optimizer.pt')