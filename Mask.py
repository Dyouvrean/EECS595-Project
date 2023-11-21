import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
# from nltk.corpus import stopwords
# nltk.download('stopwords')
from transformers import AutoTokenizer

class TextMaskingDataset(Dataset):
    def __init__(self, file_path,tokenizer, mask_token="<mask>", mask_probability=0.15 ):
        self.corpus = self.read_file(file_path)
        # self.sw_nltk = stopwords.words('english')
        self.mask_token = mask_token
        self.mask_probability = mask_probability
        self.tokenizer = tokenizer
    def read_file(self, file):
        with open(file, encoding='utf-8') as f:
            corpus = [line.strip("\n") for line in f.readlines()]
        return corpus

    def mask_sentence(self, sentence):
        print(sentence)
        words = sentence.split()
        masked_words = []
        idx = random.randint(0, len(words)-1)
        indice = []
        origin=[]
        for index,word in enumerate(words):
            word = word.strip(",.!;:\n\t")
            if index == idx:
                masked_words.append(self.mask_token)
                indice.append(True)
            else:
                masked_words.append(word)
                indice.append(False)
            origin.append(word)
        return " ".join(masked_words), " ".join(origin), indice

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        sentence = self.corpus[idx]
        masked_sentence,origin, mask_indice = self.mask_sentence(sentence)
        masked = self.tokenizer(masked_sentence, return_tensors="pt")
        origin = self.tokenizer(origin, return_tensors="pt")
        return masked, origin, mask_indice




tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# # Example usage
dataset = TextMaskingDataset(file_path="ARC_Corpus.txt",tokenizer= tokenizer)[1]
print(dataset)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=False,collate_fn=collate_fn)
# for i, batch in enumerate(dataloader):
#     print(batch)
#     if i == 1:  # Stop after retrieving 2 batches
#         break