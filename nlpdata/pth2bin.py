import torch
from transformers import GPT2ForSequenceClassification

load_path = "/home/qing.xiang/algorithm/CVData/work/best.pt"
save_path = "/home/qing.xiang/algorithm/CVData/work/best.bin"

model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
model_dict = torch.load(load_path)
with open(save_path, 'wb') as fp:
    weight_count = 0
    num=1
    for k, v in model_dict.items():
        print(k,num)
        num=num+1
        if 'num_batches_tracked' in k:
            continue
        v = v.cpu().numpy().flatten()
        for d in v:
            fp.write(d)
            weight_count+=1
    print('model_weight has Convert Completely!',weight_count)