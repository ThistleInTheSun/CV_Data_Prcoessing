import lzma
import os
import shutil

from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
from transformers import GPT2Tokenizer

data_dir = "/home/qing.xiang/data/openwebtext"
txt_dir = "/home/qing.xiang/data/openwebtext_txt"
min_length = 128
tokenized_dir = "/home/qing.xiang/data/openwebtext_tokenized"


def unzip_xz():
    os.makedirs(txt_dir, exist_ok=True)
    file_list = sorted(os.listdir(data_dir))
    for idx, file_name in tqdm(enumerate(file_list)):
        file_path = os.path.join(data_dir, file_name)
        save_path = os.path.join(txt_dir, file_name.replace(".xz", ".txt"))
        with lzma.open(file_path, 'rb') as data:
            data = data.read()
            data = str(data, encoding="utf-8")
            with open(save_path, "w") as f:
                f.write(data)
        if idx >= 100:
            raise


def train_merges():
    txt_list = [os.path.join(txt_dir, file) for file in os.listdir(txt_dir)]
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(txt_list, vocab_size=20000)
    tokenizer.save_model(".")


def build_files(data_path, tokenized_data_path, full_tokenizer, min_length):
    with open(data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        lines = [line.replace('\n', ' [SEP] ') for line in lines]  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束
    all_len = len(lines)
    sublines = lines
    sublines = [full_tokenizer.tokenize(line) for line in sublines if
                len(line) > min_length]  # 只考虑长度超过min_length的句子
    sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]
    full_line = []
    for subline in sublines:
        full_line.append(full_tokenizer.convert_tokens_to_ids('[MASK]'))  # 文章开头添加MASK表示文章开始
        full_line.extend(subline)
        full_line.append(full_tokenizer.convert_tokens_to_ids('[CLS]'))  # 文章之间添加CLS表示文章结束
    with open(tokenized_data_path, 'w') as f:
        for id in full_line:
            f.write(str(id) + ' ')
    print('finish')



tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
txt_list = sorted(os.listdir(txt_dir))
os.makedirs(tokenized_dir, exist_ok=True)
for file in txt_list:
    file_path = os.path.join(txt_dir, file)
    tokenized_data_path = os.path.join(tokenized_dir, file)
    build_files(file_path, tokenized_data_path, tokenizer, min_length)