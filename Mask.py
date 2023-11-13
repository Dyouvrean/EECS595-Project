import pandas as pd
import random
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')




def read_file(file):
    with open(file, encoding='utf-8') as f:
         corpus = []
         for line in f.readlines():
             corpus.append(line.strip("\n"))
    return corpus


def sw_remove(row,sw):
    res=[]
    row = row.split(" ")
    for word in row:
        if word.lower() not in sw:
            res.append(word)
    return " ".join(res)




def mask_sentence(sentence, sw,mask_token="<mask>", mask_probability=0.15):
    words = sentence.split()
    masked_words =[]
    for word in words:
        word= word.strip(",.\n\t")
        if random.random()<mask_probability and word.lower() not in sw:
            masked_words.append(mask_token)
        else:
            masked_words.append(word)
    return " ".join(masked_words)



corpus= read_file("ARC_Corpus.txt")[:200]
corpus= pd.DataFrame(corpus, columns=['sentence'])
sw_nltk = stopwords.words('english')
#corpus["sw_removed"] = corpus.apply(lambda row: sw_remove(row['sentence'],sw_nltk),axis=1)
corpus["masked"]=corpus.apply(lambda row: mask_sentence(row['sentence'],sw_nltk),axis=1)
print(corpus.head())