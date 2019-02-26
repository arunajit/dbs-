import datetime, re, sys
from sklearn.feature_extraction.text import TfidfVectorizer

def tokenize_and_stem(train):
    tokens = [word for sent in nltk.sent_tokenize(train) for word in nltk.word_tokenize(train)]
    filtered_tokens = []
    
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

        
tfidf = TfidfVectorizer(tokenizer=tokenize_and_stem, stop_words='english', decode_error='ignore')
sys.stdout.flush()
tdm = tfidf.fit_transform(train.values()) 



def extract_entities(train):
	entities = []
  for x in train:
    for sentence in nltk.sent_tokenize(x):
        chunks = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence)))
        entities.extend([chunk for chunk in chunks if hasattr(chunk, 'node')])
    return entities

for entity in extract_entities(train):
    print ('[' + entity.node + '] ' + ' '.join(c[0] for c in entity.leaves()))


#still under progress......
