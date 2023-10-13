import glob
import json
import os
import pickle
from tqdm import tqdm

from spacy.lang.en import English
from spacy.lang.hi import Hindi

from collections import Counter
import pickle


# Configration
ENG_DATASET = "/Users/ppuser/kllama/llama2.c/data/tinystories/"
ENG_SHARDS = glob.glob(ENG_DATASET+"*.json")

HIN_DATASET = "/Users/ppuser/kllama/llama2.c/data/translated_data/"
HIN_SHARDS = glob.glob(HIN_DATASET+"*.json")

hindi_stories={}
for hs in tqdm(HIN_SHARDS):
    stories=pickle.load(open(hs,"rb"))
    for k in stories:
        story=stories[k]["data"]["translations"][0]["translatedText"].replace("&quot;",'"')
        hindi_stories[k]=story

os.makedirs("/Users/ppuser/kllama/hinllama/data/hindi_dataset",exist_ok=True)
with open("/Users/ppuser/kllama/hinllama/data/hindi_dataset/stories.json","w") as f:
    f.write(json.dumps(hindi_stories))

HIN_DATASET = "/Users/ppuser/kllama/hinllama/data/hindi_dataset/stories.json"


nlp_en = English()
tok_en = nlp_en.tokenizer

nlp_hi = Hindi()
toke_hi = nlp_hi.tokenizer

vocab_en=Counter()
vocab_hi=Counter()

# Find unique words in Hindi corpus using spacy tokenizer
for idx in tqdm(hindi_stories,leave=False,colour="green"):
    v=[]
    story=hindi_stories[idx]
    doc=toke_hi(story)
    vocab=[x.text for x in doc]
    vocab_hi.update(vocab)

pickle.dump(vocab_hi,open("/Users/ppuser/kllama/hinllama/models/vocab_hi.bin","wb"))

# Find unique words in English corpus using spacy tokenizer
for shard in tqdm(ENG_SHARDS,leave=False,colour="green"):
    stories=json.load(open(shard,"r"))
    for story in tqdm(stories):
        doc=tok_en(story['story'])
        vocab=[x.text for x in doc]
        vocab_en.update(vocab)

pickle.dump(vocab_en,open("/Users/ppuser/kllama/hinllama/models/vocab_en.bin","wb"))
