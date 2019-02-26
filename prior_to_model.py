import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from textblob import Word
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import string


dataset = pd.read_csv('./output_file.csv')
df = pd.DataFrame(dataset)
train,test,train_head,test_head=train_test_split(df['Content'], df['FileName'],test_size=0.2)
labels  =['employment','amendment']

#punctuation
train['Content'] = train.apply(lambda x: " ".join(x.lower() for x in x.translate(string.maketrans("",""), string.punctuation) ))
train['Content'].head()

#lowercase
train['Content'] = train.apply(lambda x: " ".join(x.lower() for x in x.split()))
train['Content'].head()

#stopwords
stop = stopwords.words('english')
train['Content'] = train['Content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
train['Content'].head()

#commonwords
freq = pd.Series(' '.join(train['Content']).split()).value_counts()[:10]
freq = list(freq.index)
train['Content'] = train['Content'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
train['Content'].head()

#Lemmatization
train['Content'] = train['Content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
train['Content'].head()

#bagofwords 
'''can be even directly applied for file name also'''
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
train_bow = bow.fit_transform(train['Content'])
print(train_bow)

bow_name = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
train_bow_name = bow.fit_transform(train['FileName'])
print(train_bow)


#Word embeddings
outp=[]
glove2word2vec(train['Content'], outp)
model = KeyedVectors.load_word2vec_format(outp, binary=False)
print((model['go'] + model['away'])/2)



#classifier
n_neighbors = 2
weights = 'uniform'
weights = 'distance'
clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors, weights=weights)

#test 
clf.fit(train_head,train)
out = clf.predict(test_head)

score = accuracy_score(train_head, out)

