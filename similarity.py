import operator
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from extract_aspect_terms import *

# load the Stanford GloVe model
filename = '~/workspace/SA/glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)

print ("len(aspect_term_list)", str(len(aspect_term_list)))

aspect_term_mapping = {}
for aspect_term in aspect_term_list: 
    try:   
        aspects_similarity = {}
        aspects_similarity["food"] = model.similarity(aspect_term, 'food')
        aspects_similarity["service"] = model.similarity(aspect_term, 'service')
        aspects_similarity["price"] = model.similarity(aspect_term, 'price')
        aspects_similarity["ambience"] = model.similarity(aspect_term, 'ambience')
        aspect_term_mapping[aspect_term] = (max(aspects_similarity.items(), key=operator.itemgetter(1))[0])
    except:
        aspect_term_mapping[aspect_term] = 'misc'

# print (aspect_term_mapping)


# f = open("weights.json", "w") 
# f.write(str(aspects_similarity))
# f.close()