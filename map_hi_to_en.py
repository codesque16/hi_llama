import json
import pickle
from tqdm import tqdm
from spacy.lang.hi import Hindi
from spacy.lang.en import English
from collections import Counter
import pickle
from sentence_transformers import SentenceTransformer, util
import torch

MAP_FILE="final_mapping.json"

model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
def map_hindi_to_english(hindi_list, english_list): 
    # Create embeddings for Hindi and English words
    hindi_embeddings = model.encode(hindi_list, convert_to_tensor=True)
    english_embeddings = model.encode(english_list, convert_to_tensor=True)
    
    # Map each Hindi word to the closest English word in terms of cosine similarity
    mapping = {}
    for h_word, h_embedding in zip(hindi_list, hindi_embeddings):
        max_similarity = float('-inf')
        mapped_word = None
        for e_word, e_embedding in zip(english_list, english_embeddings):
            similarity = util.pytorch_cos_sim(h_embedding, e_embedding).item()
            if similarity > max_similarity:
                max_similarity = similarity
                mapped_word = e_word
        mapping[h_word] = mapped_word

    return mapping

def generate_stem_words(word,buffer=1):
    suffixes = {
        1: [u"ो",u"े",u"ू",u"ु",u"ी",u"ि",u"ा"],
        2: [u"कर",u"ाओ",u"िए",u"ाई",u"ाए",u"ने",u"नी",u"ना",u"ते",u"ीं",u"ती",u"ता",u"ाँ",u"ां",u"ों",u"ें"],
        3: [u"ाकर",u"ाइए",u"ाईं",u"ाया",u"ेगी",u"ेगा",u"ोगी",u"ोगे",u"ाने",u"ाना",u"ाते",u"ाती",u"ाता",u"तीं",u"ाओं",u"ाएं",u"ुओं",u"ुएं",u"ुआं"],
        4: [u"ाएगी",u"ाएगा",u"ाओगी",u"ाओगे",u"एंगी",u"ेंगी",u"एंगे",u"ेंगे",u"ूंगी",u"ूंगा",u"ातीं",u"नाओं",u"नाएं",u"ताओं",u"ताएं",u"ियाँ",u"ियों",u"ियां"],
        5: [u"ाएंगी",u"ाएंगे",u"ाऊंगी",u"ाऊंगा",u"ाइयाँ",u"ाइयों",u"ाइयां"],
    }
    
    for L in suffixes:
        if len(word) > L + buffer:
            for suf in suffixes[L]:
                #print type(suf),type(word),word,suf
                if word.endswith(suf):
                    #print 'h'
                    return word[:-L]
    return word

mcw_map=[
 ('।', '.'),
 ('और', 'and'),
 (',', ','),
 ('"', '"'),
 ('.', '.'),
 ('!', '!'),
 ('-', '-')
]

nlp_en = English()
nlp_hi = Hindi()

tok_hi = nlp_hi.tokenizer
tok_en = nlp_en.tokenizer

nlp_en.add_pipe('sentencizer')
nlp_hi.add_pipe('sentencizer')

vocab_hi=pickle.load(open("./models/vocab_hi.bin","rb"))
vocab_en=pickle.load(open("./models/vocab_en.bin","rb"))

eng_stories=json.load(open("/Users/ppuser/kllama/tinystories/stories_unique_50.json"))
hin_stories=json.load(open("/Users/ppuser/kllama/hinllama/data/hindi_dataset/stories.json","r"))

print(f"""
English vocabulary size: {len(vocab_en)}
Hindi vocabulary size: {len(vocab_hi)}
""")

class Kun:
    def __init__(self):
        self.maps={}
        self.docs={}
    
    def update(self,k,v):
        if k not in self.maps:
            self.maps[k]=Counter()
            self.docs[k]=0
        self.maps[k].update(set(v))
        self.docs[k]= self.docs[k]+1

# Pick stories which only have same number of sentences in Hindi and English =====

mapping_dict_en=Kun()
mapping_dict_hi=Kun()
consistent_ids=[]
eng_tokens_idf=Counter()
hi_tokens_idf=Counter()
for idx in tqdm(hin_stories):
    hi_story=hin_stories[idx]
    en_story=eng_stories[idx]
    
    en_sentences=[x.text for x in list(nlp_en(en_story).sents)]
    hi_sentences=[x.text for x in list(nlp_hi(hi_story).sents)]

    if len(en_sentences)==len(hi_sentences):
        consistent_ids.append(idx)
        for e,h in zip(en_sentences,hi_sentences):
            etokens=[x.text for x in tok_en(e) if x.text not in [y[1] for y in mcw_map]]
            htokens=[generate_stem_words(x.text) for x in tok_hi(h) if x.text not in [y[0] for y in mcw_map]]
            for t_ in htokens:
                mapping_dict_hi.update(t_,etokens)
            for t_ in etokens:
                mapping_dict_en.update(t_,htokens)
            eng_tokens_idf.update(list(set(etokens)))
            hi_tokens_idf.update(list(set(htokens)))

# ****************
print(f"Vocabulary to be covered {len(hi_tokens_idf)}")
# ========= Create list of most common words from english to hindi and hindi to english

truncated_dict_en={}
norm_en={}
for k in tqdm(mapping_dict_en.maps):
    truncated_dict_en[k]=mapping_dict_en.maps[k].most_common(20)
    norm_en[k]=[(x[0],x[1]*1./hi_tokens_idf[x[0]]) for x in truncated_dict_en[k]]
    norm_en[k]=[x for x in sorted(norm_en[k],key=lambda x: -x[1])[:15] if x[1]>0.3]

truncated_dict_hi={}
norm_hi={}
for k in tqdm(mapping_dict_hi.maps):
    truncated_dict_hi[k]=mapping_dict_hi.maps[k].most_common(20)
    norm_hi[k]=[(x[0],x[1]*1./eng_tokens_idf[x[0]]) for x in truncated_dict_hi[k]]
    norm_hi[k]=[x for x in sorted(norm_hi[k],key=lambda x: -x[1])[:15] if x[1]>0.3]

# ****************

# Start creating final mapping
"""
Logic:
For every hindi word start parsing the corresponding english words
Choose the top two words . If the count of the top is 1.5x the second map the 
first word to the hindi word
Add the english word to the eng list and hindi word to the hindi list
"""
final_mapping={}
eng_words=set()
hi_words=set()
prev_length=0
stem={}
while True:
    for h in tqdm(truncated_dict_hi):
        if h in hi_words:
            continue
        for x in sorted(truncated_dict_hi[h],key=lambda a: -a[1]):
            if x[0] in eng_words:
                continue
            rev_=truncated_dict_en[x[0]]
            f_=0
            t_=[]
            for y in rev_:
                if y[0] in hi_words:
                    continue
                f_+=1
                t_.append(y)
                if f_==2:
                    break

            if len(t_)>0 and t_[0][0]==h:
                if len(t_)==1 or (len(t_)>1 and t_[0][1]>(t_[1][1]*1.5)):
                    final_mapping[h]=x[0]
                    hi_words.add(h)
                    eng_words.add(x[0])
                    break

    if len(hi_words)>prev_length:
        prev_length=len(hi_words)
    else:
        break

print(f"Vocabulary covered ... {len(final_mapping)}")
# Stem words :START ===============
hi_not_covered={}
for x in hi_tokens_idf:
    if x not in hi_words:
        stem_=generate_stem_words(x,2)
        if stem_ in hi_not_covered:
            hi_not_covered[stem_].append(x)
        else:
            hi_not_covered[stem_]=[x]

hi_covered={}
for x in hi_words:
    stem_=generate_stem_words(x,2)
    if stem_ in hi_covered:
        hi_covered[stem_].append(x)
    else:
        hi_covered[stem_]=[x]

for x in list(set(hi_not_covered.keys()).intersection(set(hi_covered.keys()))):
    for h in hi_not_covered[x]:
        eng_word_=sorted(hi_covered[x],key=lambda a: len(a))[0]
        final_mapping[h]= final_mapping[eng_word_]
        hi_words.add(h)
        eng_words.add(eng_word_)
        stem[h]=1

# Stem words : =============== END
print(f"Vocabulary covered ... {len(final_mapping)}")


prev_length=len(hi_words)
while True:
    for h in tqdm(truncated_dict_hi):
        if h in hi_words:
            continue
        for x in sorted(truncated_dict_hi[h],key=lambda a: -a[1]):
            if x[0] in eng_words:
                continue
            rev_=truncated_dict_en[x[0]]
            f_=0
            t_=[]
            for y in rev_:
                if y[0] in hi_words:
                    continue
                f_+=1
                t_.append(y)
                if f_==2:
                    break

            if len(t_)>0 and t_[0][0]==h:
                if len(t_)==1 or (len(t_)>1 and t_[0][1]>(t_[1][1]*1.5)):
                    final_mapping[h]=x[0]
                    hi_words.add(h)
                    eng_words.add(x[0])
                    break

    if len(hi_words)>prev_length:
        prev_length=len(hi_words)
    else:
        break

print(f"Vocabulary covered ... {len(final_mapping)}")
# Stem words :START ===============
hi_not_covered={}
for x in hi_tokens_idf:
    if x not in hi_words:
        stem_=generate_stem_words(x,2)
        if stem_ in hi_not_covered:
            hi_not_covered[stem_].append(x)
        else:
            hi_not_covered[stem_]=[x]

hi_covered={}
for x in hi_words:
    stem_=generate_stem_words(x,2)
    if stem_ in hi_covered:
        hi_covered[stem_].append(x)
    else:
        hi_covered[stem_]=[x]
        
for x in list(set(hi_not_covered.keys()).intersection(set(hi_covered.keys()))):
    for h in hi_not_covered[x]:
        final_mapping[h]= final_mapping[sorted(hi_covered[x],key=lambda a: len(a))[0]]
        stem[h]=1

# Stem words : =============== END
print(f"Vocabulary covered ... {len(final_mapping)}")


eng_words=set([x[1] for x in final_mapping.items()])
hi_words=set([x[0] for x in final_mapping.items()])


# Start analyzing remainder words and create cross reference for hindi and english

not_covered_hi={}
not_covered_en={}
for k in tqdm(mapping_dict_en.maps):
    if k in eng_words:
        continue
    not_covered_en[k]=[x for x in mapping_dict_en.maps[k].items() if x[0] not in hi_words]
    not_covered_en[k]=sorted(not_covered_en[k],key=lambda x: -x[1])

for k in tqdm(mapping_dict_hi.maps):
    if k in hi_words:
        continue
    not_covered_hi[k]=[x for x in mapping_dict_hi.maps[k].items() if x[0] not in eng_words]
    not_covered_hi[k]=sorted(not_covered_hi[k],key=lambda x: -x[1])

count=0
coverage={}
priority=[]
for idx,k in tqdm(enumerate(not_covered_hi),total=len(not_covered_hi)):
    a_=[(x[0],x[1]*1.0/hi_tokens_idf[k]) for x in not_covered_hi[k] if x[0] not in eng_words and x[1]*1.0/hi_tokens_idf[k] >0.1]
    if len(a_)>0 and a_[0][1]>0.4:
        #print(k)
        s_=generate_stem_words(k,buffer=1)
        a__=[]
        for b_ in a_:
            score=b_[1]*\
                sum([c_[1]/eng_tokens_idf[b_[0]] for c_ in not_covered_en[b_[0]] if generate_stem_words(c_[0])==s_])
            a__.append((b_[0],score))
        a__=sorted(a__,key=lambda c: -c[1])
        coverage[k]=a__
        priority.append((k,a__[0][1] if len(a__)>0 else 0))
        #print(a__)
        count+=1

priority=sorted(priority,key=lambda c: -c[1])
stage2={}
for k,_ in tqdm(priority):
    if k in hi_words:
        continue
    for w in coverage[k]:
        if w[0] in eng_words:
            continue
        final_mapping[k]=w[0]
        stage2[k]=w[0]
        eng_words.add(w[0])
        hi_words.add(k)
        break

print(f"Vocabulary covered ... {len(final_mapping)}")
# Stem words :START ===============
hi_not_covered={}
for x in hi_tokens_idf:
    if x not in hi_words:
        stem_=generate_stem_words(x,1)
        if stem_ in hi_not_covered:
            hi_not_covered[stem_].append(x)
        else:
            hi_not_covered[stem_]=[x]

hi_covered={}
for x in hi_words:
    stem_=generate_stem_words(x,1)
    if stem_ in hi_covered:
        hi_covered[stem_].append(x)
    else:
        hi_covered[stem_]=[x]
        
for x in list(set(hi_not_covered.keys()).intersection(set(hi_covered.keys()))):
    for h in hi_not_covered[x]:
        final_mapping[h]= final_mapping[sorted(hi_covered[x],key=lambda a: len(a))[0]]
        stem[h]=1

print(f"Vocabulary covered ... {len(final_mapping)}")

for k in mcw_map:
    final_mapping[k[0]]=k[1]

print(f"Vocabulary covered ... {len(final_mapping)}")

# Write all files
pickle.dump(mapping_dict_hi,open("kunfile_hi.bin","wb"))
pickle.dump(mapping_dict_en,open("kunfile_en.bin","wb"))

pickle.dump(hi_tokens_idf,open("idf_hi.bin","wb"))
pickle.dump(eng_tokens_idf,open("idf_en.bin","wb"))

with open(MAP_FILE,"w") as f:
    f.write(json.dumps(final_mapping,indent=4))

with open("./stem.json","w") as f:
    f.write(json.dumps(stem,indent=4))

with open("./stage2.json","w") as f:
    f.write(json.dumps(stage2,indent=4))

pickle.dump(consistent_ids,open("consistent_stories.bin","wb"))

