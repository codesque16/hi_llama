import numpy
import pickle
import tqdm

def load_vecs(path, limit_no_words=None):

    words = []
    vectors = []
    w2v={}

    with open(path, 'r') as fp:
        lines = len(fp.readlines())
    with open(path,'r') as fp:
        #lines_raw=fp.readlines()
        for ix, line in tqdm.tqdm(enumerate(fp),total=lines):
            if ix == 0:
                continue
            line = line.split()
            word = line[0]
            if len(line[1:]) == 300:
                #words.append(word)
                #vectors.append(line[1:])
                w2v[word]=numpy.array(line[1:], dtype=float)
            else:
                print(word)
            if limit_no_words and len(w2v.keys()) == limit_no_words:
                break

    #return words, numpy.array(vectors, dtype=float)
    return w2v

hi_w2v = load_vecs("/Users/ppuser/kllama/hllm/data/wiki.hi.align.vec")
eng_w2v = load_vecs("/Users/ppuser/kllama/hllm/data/wiki.en.align.vec")

pickle.dump(hi_w2v,open("./models/w2v_hi.bin","wb"))
pickle.dump(eng_w2v,open("./models/w2v_en.bin","wb"))