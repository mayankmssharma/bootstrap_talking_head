# coding: utf-8

# In[197]:
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from sklearn.model_selection import train_test_split
import pickle


# In[31]:


books_data_base = pd.read_csv('data/booksoutnew1.csv')
data_frame = pd.read_csv('data/augmented_data_set.csv')


# In[32]:


books_data_base.head()


# In[33]:


def tokenize(text) :
    

    stop_words = set(stopwords.words('english'))

    #soup = BeautifulSoup(text)
    #content = soup.get_text()
    
    raw_tokens = word_tokenize(text)
    raw_tokens = [w.lower() for w in raw_tokens]
    
    #tokens = list()
    
    #for tk in raw_tokens :
    #    tkns = tk.split('-')
    #    for tkn in tkns :
    #        tokens.append(tkn)
    
    #old_punctuation = string.punctuation
    #new_punctuation = old_punctuation.replace('-','')
    
    #table = str.maketrans('','',new_punctuation)
    table = str.maketrans('','',string.punctuation)
    
    stripped = [w.translate(table) for w in raw_tokens]
    
    words = [word for word in stripped if word.isalpha()]
    words = [w for w in words if not w in stop_words]
    
    return words


# In[34]:


train_set = set()
for index,row in data_frame.iterrows() :
    #print(' String : {}\t Label : {}'.format(row['text'],row['label']))
    train_set.add(tuple([row['Utterance'],row['Label']]))


# In[35]:


train_list = list(train_set)
print(train_list[:5])


# In[36]:


train_list = list(train_set)
print(train_list[:5])


# In[37]:


raw_train_data = list()
raw_train_labels = list()
max_sentence_size = -9999
vocab_set = set()

for train_tuple in train_list :
    raw_input, label = train_tuple
    raw_tokens = tokenize(raw_input)
    raw_train_data.append(raw_tokens)
    max_sentence_size = max(max_sentence_size,len(raw_tokens))
    for tk in raw_tokens :
        vocab_set.add(tk)
    raw_train_labels.append(label)
vocab_size = len(vocab_set)


# In[38]:


word_idx = dict()
for index, word in enumerate(sorted(vocab_set)) :
    word_idx[word] = index


# In[39]:


def vectorize(tokenized_line,max_sentence_size,word_idx) :
    lq = max(0,max_sentence_size-len(tokenized_line))
    vec_line = [word_idx[w] if w in word_idx.keys() else 0 for w in tokenized_line] + lq*[0]
    vec_line = np.array(vec_line)
    return vec_line


# In[40]:


train_data = list()
train_labels = list()
for i in range(len(raw_train_data)) :
    #print(raw_train_data[i])
    vec_train_set = vectorize(raw_train_data[i],max_sentence_size,word_idx)
    vec_label = raw_train_labels[i]
    train_data.append(vec_train_set)
    train_labels.append(vec_label)

train_x = np.array(train_data)
train_y = np.array(train_labels)
train_labels_one_hot = to_categorical(train_labels)
train_y_one_hot = np.array(train_labels_one_hot)


# ## Using Deep Learning Approach

# In[41]:


x_train,x_test, y_train, y_test = train_test_split(train_x, train_y_one_hot, test_size=0.01, random_state=10)


# In[42]:


embedding_size = 128
deep_net = Sequential()
deep_net.add(Embedding(vocab_size,embedding_size,input_length=max_sentence_size,mask_zero=True))
deep_net.add(LSTM(embedding_size))
deep_net.add(Dense(4,activation='softmax'))
deep_net.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(deep_net.summary())


# In[43]:


deep_net.fit(x_train,y_train,epochs=50)


# In[44]:


#deep_net.evaluate(x_train,y_train)


# In[45]:


#deep_net.evaluate(x_test,y_test)


# In[199]:

knowledge_base = dict()
for index, row in books_data_base.iterrows() :
    tk_words = tokenize(row['book'])
    for word in tk_words :
        if word not in knowledge_base.keys() :
            book_list = list()
        else :
            book_list = knowledge_base[word]
        book_list.append(row['book'])
        knowledge_base[word] = book_list

model_file = open('model_data/deep_net_model.pkl','wb')
pickle.dump([deep_net,max_sentence_size,word_idx],model_file)
model_file.close()
