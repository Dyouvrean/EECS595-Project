import argparse
import os
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from datasets import load_dataset, load_metric
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

from transformers import GPT2Tokenizer, GPT2Model

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
batch_size = 8



def tokenize_function(examples, tokenizer):
    first_sentences = [[context] for context in examples["question"]]
    second_sentences = [i['text'] for i in examples["choices"]]
    label = [i['label'] for i in examples["choices"]]
    first_sentences = sum(first_sentences, [])
    combine = [zip(s, f) for f, s in zip(second_sentences, label)]
    new = []
    for i in combine:
        pair = []
        for j in i:
            pair.append((".".join(j)))
        new.append(pair)

    combine = [["{} {}".format(f, " ".join(s)) + " Answer:"] for f, s in zip(first_sentences, new)]
    combine = sum(combine, [])
    tokenized_examples = tokenizer(combine, truncation=True)
    return tokenized_examples


def convert(dataset):
    labels = dataset['answerKey']
    res = []
    for label in labels:
        if label not in ['A', 'B', 'C', 'D','E']:
            res.append(int(label)-1)
        else:
            res.append(ord(label) - 65)
    dataset['answerKey'] = torch.tensor(res)
    return dataset


# dataset = load_dataset("ai2_arc", 'ARC-Challenge')
# tokenized_datasets = dataset.map(convert_label, batched=True)


# train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=batch_size)
# eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=batch_size)
# test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=batch_size)

# # Assuming tokenized_datasets is already prepared and contains the required features

# # Create DataLoader for the 'train' split
# # Iterate over each batch and feed it to the model
# outputs = []
# for batch in train_dataloader:
#     # Prepare input tensors. You might need to adjust keys based on your dataset.
#     batch = tokenizer.pad(batch,padding= True,return_tensors="pt")
#     inputs = {key: value for key, value in batch.items() if key in ['input_ids', 'attention_mask']}

#     # Forward pass
#     output = model(**inputs)
#     outputs.append(output)
#     # process the output as needed


