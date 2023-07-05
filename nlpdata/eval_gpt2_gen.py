import jsonlines
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from transformers import (GPT2LMHeadModel, GPT2Model, GPT2Tokenizer, pipeline,
                          set_seed)


def read_jsonl(test_txt):
    data = []
    with open(test_txt, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            data.append(item)
    return data


def get_wsc_prompt(info):
    sentence = info["text"]
    A = info["target"]["span1_text"]
    B = info["target"]["span2_text"]
    prompt = "'{}'. In this sentence, Do '{}' and '{}' refer to the same target?".format(sentence, A, B)
    idx = info["idx"]
    label = int(info["label"])
    label = bool(label)
    return prompt, idx, label, sentence


def get_cola_prompt(info):
    sentence = info["text"]
    idx = info["idx"]
    label = int(info["label"])
    label = bool(label)
    prompt = "'{}'. Is this sentence acceptable or unacceptable? This sentence is".format(sentence)
    return prompt, idx, label, sentence


def get_sst_prompt(info):
    sentence = info["text"]
    idx = info["idx"]
    label = int(info["label"])
    label = bool(label)
    # Is the sentiment of the sentence positive or negative? This sentence is
    prompt = "'{}'. The sentiment of this review is ".format(sentence)
    return prompt, idx, label, sentence


if __name__ == "__main__":
    model_name = "gpt2"
    data_name = "wsc"
    test_txt = "./work/tasks/data/{}/val.jsonl".format(data_name)
    data = read_jsonl(test_txt)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    print("\n\n\nmodel: {}, data: {}".format(model_name, data_name))

    right = 0
    coutF = 0
    for i, info in enumerate(data):
        if data_name == "wsc":
            prompt, idx, label, sentence = get_wsc_prompt(info)
            A, B = "yes", "no"
        elif data_name == "cola":
            prompt, idx, label, sentence = get_cola_prompt(info)
            A, B = "acceptable", "unacceptable"
        elif data_name == "sst":
            prompt, idx, label, sentence = get_sst_prompt(info)
            A, B = "positive", "negative"
        else:
            raise ValueError("error data_name '{}'".format(data_name))
        
        encoded_input = tokenizer(prompt, return_tensors='pt')
        outputs = model(**encoded_input)
        next_token_logits = outputs[0][0, -1, :]
        
        Ascore = next_token_logits[tokenizer.convert_tokens_to_ids("Ġ" + A)]
        Bscore = next_token_logits[tokenizer.convert_tokens_to_ids("Ġ" + B)]
        pre = Ascore >= Bscore

        if i % 20 == 0:
        # if pre != label:
            print("\nprompt:", prompt)
            print("{}: {}\t {}: {}\t pre:{}\t label:{}\t cout:{}".format(
                A, Ascore, 
                B, Bscore, pre, label, pre == label))
        if pre == False:
            coutF += 1
        right += 1 if pre == label else 0
    print(right / len(data), "total: {} / {}".format(right, len(data)))
    print("False cout:", coutF)
