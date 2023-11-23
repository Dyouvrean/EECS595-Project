import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
# from nltk.corpus import stopwords
# nltk.download('stopwords')
from transformers import AutoTokenizer

class TextNoisyDataset(Dataset):
    def __init__(self, file_path,tokenizer, mask_probability=0.15 ):
        self.corpus = self.read_file(file_path)
        # self.sw_nltk = stopwords.words('english')
        self.tokenizer = tokenizer
    def read_file(self, file):
        with open(file, encoding='utf-8') as f:
            corpus = [line.strip("\n") for line in f.readlines()]
        return corpus[:10]
    
    def introduce_typo(self,word):
        if len(word) > 1:
            char_pos = random.randint(0, len(word) - 1)
            word = word[:char_pos] + word[char_pos+1:]
        return word





    def add_noisy(self, sentence):
        print(sentence)
        words = sentence.split()
        noisy_words = []
        idx = random.randint(0, len(words)-1)
        indice = []
        origin=[]
        for index,word in enumerate(words):
            word = word.strip(",.!;: \n\t")
            if index == idx:
                noisy_words.append(self.introduce_typo(word))
                indice.append(True)
            else:
                noisy_words.append(word)
                indice.append(False)
            origin.append(word)
        noisy_words.append("please reconstruct the sentence by changing the mistake word.")
        return " ".join(noisy_words), " ".join(origin), indice

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        sentence = self.corpus[idx]
        masked_sentence,origin, mask_indice = self.add_noisy(sentence)
        masked = self.tokenizer(masked_sentence, return_tensors="pt",padding=True, truncation=True)
        origin = self.tokenizer(origin, return_tensors="pt",padding=True, truncation=True)
        return masked, origin, mask_indice
    
    
    




# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# # # Example usage
# dataset = TextMaskingDataset(file_path="ARC_Corpus.txt",tokenizer= tokenizer)[1]
# print(dataset)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=False,collate_fn=collate_fn)
# for i, batch in enumerate(dataloader):
#     print(batch)
#     if i == 1:  # Stop after retrieving 2 batches
#         break