import openai
import pickle
import tqdm

OAI_KEY="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"

vocab_hi=pickle.load(open("./models/vocab_hi.bin","rb"))
vocab_en=pickle.load(open("./models/vocab_en.bin","rb"))

print(f"""
English vocabulary size: {len(vocab_en)}
Hindi vocabulary size: {len(vocab_hi)}
""")

emb_hi={}
token_count=0
count=0
for h in tqdm.tqdm(vocab_hi,desc=f"Tokens {token_count}"):
    emb_ = openai.Embedding.create(model="text-embedding-ada-002", input=h,api_key=OAI_KEY)
    emb_hi[h]=emb_
    token_count+=emb_['usage']['total_tokens']
    count+=1
    if count>20:
        break

pickle.dump(emb_hi,open("./models/emb_hi.bin","wb"))

count=0
emb_en={}
for e in tqdm.tqdm(vocab_en):
    emb_ = openai.Embedding.create(model="text-embedding-ada-002", input=e,api_key=OAI_KEY)
    emb_en[e]=emb_
    token_count+=emb_['usage']['total_tokens']
    count+=1
    if count>20:
        break

pickle.dump(emb_en,open("./models/emb_en.bin","wb"))