import json
import pickle
from tqdm import tqdm
from spacy.lang.hi import Hindi
from spacy.lang.en import English
from collections import Counter
import pickle
from indicate import transliterate

MAP_FILE="final_mapping.json"

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

eng_words=set([x[1] for x in final_mapping.items()])
hi_words=set([x[0] for x in final_mapping.items()])

print(f"Vocabulary covered ... {len(final_mapping)}")

for k in mcw_map:
    final_mapping[k[0]]=k[1]

print(f"Vocabulary covered ... {len(final_mapping)}")

eng_words=set([x[1] for x in final_mapping.items()])
hi_words=set([x[0] for x in final_mapping.items()])

all_vocab=set(mapping_dict_hi.maps.keys())
not_mapped=all_vocab.difference(set(hi_words))
remove_common=Counter()
for k in not_mapped:
    remove_common.update([x[0] for x in mapping_dict_hi.maps[k].most_common(10)])

remove_common=[x[0].lower() for x in remove_common.most_common(50)]
b_={}
for k in not_mapped:
    b_[k]=[x for x in mapping_dict_hi.maps[k].most_common(20) if x[0].lower() not in remove_common]

stage3={}
word_counter=Counter()
for k in b_:
    if len(b_[k])>0 and b_[k][0][1]>10:
        if len(b_[k])==1 or (b_[k][0][1] > b_[k][1][1]*1.5):  
            if b_[k][0][0] in eng_words:
                word_counter.update(b_[k][0][0])
            idx=word_counter[b_[k][0][0]]
            if idx:
                stage3[k]=b_[k][0][0]+"_"+ str(idx) 
                final_mapping[k]=b_[k][0][0]+"_"+ str(idx) 
            else:
                stage3[k]=b_[k][0][0]
                eng_words.add(b_[k][0][0])
                final_mapping[k]=b_[k][0][0]


eng_words=set([x[1] for x in final_mapping.items()])
hi_words=set([x[0] for x in final_mapping.items()])

all_vocab=set(mapping_dict_hi.maps.keys())
not_mapped=all_vocab.difference(set(hi_words))

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

with open("./stage3.json","w") as f:
    f.write(json.dumps(stage3,indent=4))

pickle.dump(consistent_ids,open("consistent_stories.bin","wb"))

stage4={}
for k in tqdm(not_mapped):
    english_translated = transliterate.hindi2english(k)
    if english_translated in eng_words:
        word_counter.update(english_translated)
    idx=word_counter[english_translated]
    if idx:
        stage4[k]=english_translated+"_"+ str(idx) 
        final_mapping[k]=english_translated+"_"+ str(idx) 
    else:
        stage4[k]=english_translated
        eng_words.add(english_translated)
        final_mapping[k]=english_translated

eng_words=set([x[1] for x in final_mapping.items()])
hi_words=set([x[0] for x in final_mapping.items()])

all_vocab=set(mapping_dict_hi.maps.keys())
not_mapped=all_vocab.difference(set(hi_words))
print(f"Vocabulary covered ... {len(final_mapping)} - Not Mapped {len(not_mapped)}")


with open("./stage4.json","w") as f:
    f.write(json.dumps(stage4,indent=4))


