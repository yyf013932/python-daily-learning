import pandas as pd
import numpy as np
import gensim
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
import matplotlib.pyplot as plt

TaggedDocument = gensim.models.doc2vec.TaggedDocument

vec_size = 200

sms = pd.read_csv('d:/resources/sms_analysis/df_use.csv', index_col=0).drop_duplicates()

sms_content_np = sms['sms'].values

train = []

for i, text in enumerate(sms_content_np):
    word_list = text.split(' ')
    l = len(word_list)
    word_list[l - 1] = word_list[l - 1].strip()
    document = TaggedDocument(word_list, tags=[i])
    train.append(document)

# model_dm = Doc2Vec(train, min_count=1, window=3, size=vec_size, sample=1e-3, negative=5,
#                    workers=4)
# model_dm.train(train, total_examples=model_dm.corpus_count, epochs=70)
# model_dm.save('d:/model_dm')
model_dm = Doc2Vec.load('d:/model_dm')

vectors = [np.array(model_dm.docvecs[z.tags[0]].reshape(1, vec_size)) for z in train]

vectors_np = np.concatenate(vectors)

from sklearn.decomposition import PCA

pca = PCA(0.95)
pca.fit(vectors_np)

pca_train = pca.transform(vectors_np)

plt.scatter(pca_train[:, 0], pca_train[:, 1], s=1)
