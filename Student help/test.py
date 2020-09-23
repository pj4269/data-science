from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import networkx as nx
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


import pandas as pd

players_df = pd.read_csv("https://tf-assets-prod.s3.amazonaws.com/tf-curric/data-science/players.csv")
'''

if (response.ok):
  #get the full data from the response
  data = response.text
  soup=BeautifulSoup(data, 'html.parser')
  #content = soup.find_all(class_="content")
  content=soup.select('content')
print(content)
'''
'''

#from sklearn.datasets import fetch_20newsgroups
#newsgroups = fetch_20newsgroups()

#from pprint import pprint
#pprint(list(newsgroups.target_names))
#from sklearn.feature_extraction.text import TfidfVectorizer

#reading in the data, this time in the form of paragraphs
#from sklearn.datasets import fetch_20newsgroups
#newsgroups = fetch_20newsgroups()

#from sklearn.datasets import fetch_20newsgroups

categs =['alt.atheism',
         'rec.autos',
         'sci.electronics',
         'sci.med',
         'sci.space',
         'soc.religion.christian',
         'talk.politics.guns',
         'talk.politics.mideast']


news_train = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'), categories = categs)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words = 'english')
news_train_tfidf = vectorizer.fit_transform(news_train.data)

vocabulary = vectorizer.get_feature_names()
'''
#print (news_train_tfidf[:1] )
#print news_train_tfidf.shape

def word_topic(tfidf,solution, wordlist):
    words_by_topic=tfidf.T * solution
    components=pd.DataFrame(words_by_topic,index=wordlist)
    return components

# Extracts the top N words and their loadings for each topic.
def top_words(components, n_top_words):
    n_topics = range(components.shape[1])
    index= np.repeat(n_topics, n_top_words, axis=0)
    topwords=pd.Series(index=index)
    for column in range(components.shape[1]):
        # Sort the column so that highest loadings are at the top.
        sortedwords=components.iloc[:,column].sort_values(ascending=False)
        # Choose the N highest loadings.
        chosen=sortedwords[:n_top_words]
        # Combine loading and index into a string.
        #chosenlist=chosen.index +" "+round(chosen,2)#.map(str) 
        chosenlist=chosen.index +" "+chosen.map(str) 
        topwords.loc[column]=chosenlist
    return(topwords)

# Number of words to look at for each topic.
n_top_words = 10
ntopics=8




from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
'''


svd= TruncatedSVD(ntopics)
lsa = make_pipeline(svd, Normalizer(copy=False))
news_train_lsa = lsa.fit_transform(news_train_tfidf)

components_lsa = word_topic(news_train_tfidf, news_train_lsa, vocabulary)

topwords=pd.DataFrame()
topwords['LSA']=top_words(components_lsa, n_top_words)
# working
#print topwords['LSA']


from sklearn.decomposition import NMF

nmf = NMF(alpha=0.0, 
          init='nndsvdar', # how starting value are calculated
          l1_ratio=0.0, # Sets whether regularization is L2 (0), L1 (1), or a combination (values between 0 and 1)
          max_iter=200, # when to stop even if the model is not converging (to prevent running forever)
          n_components=ntopics, 
          random_state=0, 
          solver='cd', # Use Coordinate Descent to solve
          tol=0.0001, # model will stop if tfidf-WH <= tol
          verbose=0 # amount of output to give while iterating
         )
news_train_nmf = nmf.fit_transform(news_train_tfidf) 

components_nmf = word_topic(news_train_tfidf, news_train_nmf, vocabulary)

topwords['NNMF']=top_words(components_nmf, n_top_words)

print topwords['NNMF']



for topic in range(ntopics):
    print('Topic {}:'.format(topic))
    print(topwords.loc[topic])

'''









#df = pd.DataFrame(data=[None,None,None],columns=['a'])
#print df.describe()

import numpy as np
from sklearn.preprocessing import power_transform
data = [[-1, 2], [3, 2], [4, 5]]

#print(power_transform(data, method='yeo-johnson'))  

df = pd.DataFrame([2,'4','aaaaa','bbbbbb',10.1,np.nan,'hi'], columns = ['numerics'])
#print df

list1 = []
for i in df.numerics:
  try: 
    i = float(i)
  except:
    pass
  if type(i) in [int, float]:
    list1.append(None)
  else:
    list1.append(i)

#print list1

#df['new_column'] = list1
#print df

#df = df['numerics'].apply(lambda x: 0 if pd.isnull(x) else 1)

#df = df['numerics'].apply(lambda x: x.fillna(0))
#2. movies['homepage'].apply(lambda x: pass if x ==0  else 1)
#print df
  
# initialize list of lists 
#data = [['tom', 10, 'hi'], ['nick', 15, np.nan], ['juli', 14, 12]] 
data = [['tom', 10, 'hi'], ['nick', 15, None], ['juli', 14, 12]]   
# Create the pandas DataFrame 
df = pd.DataFrame(data, columns = ['Name', 'Age', 'test']) 
  
# print dataframe. 
#df = df[['Name', 'Age']].dropna()
#print df

url = "https://raw.githubusercontent.com/Thinkful-Ed/big-data-student-resources/master/datasets/UCI_HAR/allData.csv"
url2 = "https://raw.githubusercontent.com/Thinkful-Ed/big-data-student-resources/master/datasets/UCI_HAR/activity_labels.csv"
#c=pd.read_csv(url)# https://github.com/Thinkful-Ed/big-data-student-resources/blob/master/datasets/UCI_HAR/allData.csv

#df = pd.read_csv(url2)

#df.to_csv('activity_labels.csv')

#processing
'''
newsgroups_paras=[]
for paragraph in list(newsgroups.target_names):
    para=paragraph#[0]
    print ('para', para)
    #removing the double-dash from all words
    para=[re.sub(r'--','',word) for word in para]
    #Forming each paragraph into a string and adding it to the list of strings.
    newsgroups_paras.append(''.join(para))
print newsgroups_paras
'''

#vectorizer = TfidfVectorizer(stop_words='english')
#newsgroups_paras_tfidf=vectorizer.fit_transform(newsgroups_paras)
#print newsgroups_paras_tfidf
'''
#Our SVD data reducer.  We are going to reduce the feature space from 1379 to 130.
svd= TruncatedSVD(130)
lsa = make_pipeline(svd, Normalizer(copy=False))
# Run SVD on the training data, then project the training data.
X_train_lsa = lsa.fit_transform(X_train_tfidf)

variance_explained=svd.explained_variance_ratio_
total_variance = variance_explained.sum()
print("Percent variance captured by all components:",total_variance*100)

#Looking at what sorts of paragraphs our solution considers similar, for the first five identified topics
paras_by_component=pd.DataFrame(X_train_lsa,index=X_train)
for i in range(5):
    print('Component {}:'.format(i))
    print(paras_by_component.loc[:,i].sort_values(ascending=False)[0:10])



'''



def user_contacts(l1):
  l2 = []
  l3 = []
  for i in l1:
    #print (i[0])
    l2.append(i[0])
    try:
      l3.append(i[1])
    except IndexError:
      l3.append(None)
  dict1 = dict(zip(l2, l3))
  return dict1
#print (user_contacts(l1))
  #print i[0], i[1]

#dict1 = {}
#dict1[0] = l1

#print (l1)


l1 = [1, 2, 4, 5, 6, 7, 4]
l2 = [2, 3, 5, 6, 7, 8 ,6]

def rmse(l1, l2):
  l3 = []
  for a, b in zip(l1, l2):
    l3.append( (a - b)**2)
  return float(sum(l3))/float(len(l3) )  ** 0.5
  

#print rmse(l1, l2)



#list1 = ['bat', 'rats', 'god', 'dog', 'good', 'cat', 'arts', 'star']



def agram(l1):

  l2 = []

  for i in l1:
    for z in l1:
      if i == z:
        pass
      elif sorted(i) == sorted(z):
        l2.append(i)
        l2.append(z)        
  return list(set(l2))




# 0, 1, 1, 2, 3, 5, 8, 13, 21

def fib(n):
  l1 = [0, 1, 1]
  if n>3:
    for i in range(n-3):
      a = sum(l1[-2::] )
      l1.append(a)
  return l1



def prime(n):
  l0 = range(2, 100)
  
  if n> 3:
    l3  = []
    for i in l0:
      #l3 = []
      l2 = []
      for z in range(2, i+1):
        if i%z == 0:
          #print i, z
          l2.append(i)
      if len(l2)<3:
        l3.append(i)
  return l3[:n]

#print prime(10)

#l1 = range(1, 100)






def test(list1):
   list3 = []
   for i in list1:
      for z in list1: 
         if i==z:
            pass
         elif sorted(i)==sorted(z):
            list3.append(i)
            list3.append(z)
         else:
            pass

   return list(set(list3))

#print test(list1)




def Anag(l1):
  l2 = []
  l3 = []

  for i in l1:
    l2.append( ''.join( sorted(i) ) )
  for k in l2:
    l4 = []
    for z in l2:
      if k == z:
        l4.append(k)
    if len(l4)>1:
      l3.append(k)
  l3 = list(set(l3))
  print l3
  l5 = []
  for a, b in zip(l2, range(1, len(l2)+1)):
    if a in l3:
      print a, b, l5.append(b)
  #print l5
  l6 = []
  for i in l5:
    #print i
    #l1[i]
    l6.append(l1[i-1])
  return l6
  
#print (Anag(list1) )




#a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
#b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
#print [i for i in a for k in b if i==k]


# fibonacci

# how many fib numbers?
def fib(num):
  l1 = [0, 1, 1]
  if num > 0:
    if num ==1:
      return l1[0]
    elif num == 2:
      return l1[:2]
    elif num == 3:
      return l1
    else:# 5
      for i in range(num-3):
        l1.append( sum(l1[-2:]) )
      return l1
  else:
    return ''
 
#print (fib(12))

# prime

def prime(list1):
  l2 = []
  for i in list1:
    if i >= 2:
      # can be divided by itself only
      div = range(2, i+1)
      l3 = []
      #print div
      for k in div:
        if i%k == 0:
          l3.append(k)
      if len(l3) ==1:      
        l2.append(i)
  return l2

#print prime(l1)

'''
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2)
kf.get_n_splits(X)
print X
print y


for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print X_train


'''
#TRAIN: [2 3] TEST: [0 1]
#TRAIN: [0 1] TEST: [2 3]

'''
print 'hi'
with open("technical coach.txt", "a") as f:
     d = f#.write("new line\n")
     print d


'''
# file writing with append 
#with open('text.txt', 'a+') as file1:
#    file1.write('hello')

'''

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
fill_list = usedu_df.select_dtypes(include=numerics)

years = usedu_df['YEAR'].unique()
print (usedu_df.isnull().sum()/len(usedu_df)) * 100

for year in years:
 usedu_df.fillna({x: usedu_df.loc[usedu_df["YEAR"] == year][x].mean() for x in fill_list}, inplace = True)

print (usedu_df.isnull().sum()/len(usedu_df)) * 100
'''
'''
1. Python: 1: done
2.  SQL, Numpy, Pandas, Sci-Kit-Learn (SK Learn) => 
3. SQL: 

2. Machine Learning: 2
3. Tensorflow: 3
4. Pyspark: 5

Python: 
numpy??

1. Problems => hard ones => hackerrank 
'''            


#- list comprehension: done
#- primality: done
# fibonacci: 
# bigram:

#list1 = ['aba', 'bbz', 'abba']
list1 = ['bat', 'rats', 'god', 'dog', 'good', 'cat', 'arts', 'star']
#print 'list1: ',list1


def t(l1):
  l2 = []

  for i in l1:
    for z in l1:
      #print i, z, sorted(z)
      if i==z:
        pass
      elif sorted(i)==sorted(z):
        l2.append(i)
        l2.append(z)
      else:
        pass
        #c = ''.join(sorted(z))
        #print sorted(i), sorted(z)
  '''
        print 'hi ', c, i
        if i==c:
          l2.append(i, z)  
          print i, z
        else:
          pass
  '''    
  return set(l2)

#print t(list1)

def anagram(l1):
  dict1 = zip(l1, range(len(l1)))
  l2 = []
  l3 = []
  for a in l1:
    l2.append(''.join(sorted(a) ) )

  dict2 = zip(l2, range(len(l2)))
  c = sorted(a) 
  #print dict2
  l4 = list(set(l2))
  l5 = []
  for i in l4:
    if i in l2:
      l5.append(i)
      print i
  print 'l5: ',l5
  #for a, b in dict2: 
  
  #for i in l2:
  #  l4 = l2
  #  l4.remove(i)
  #  print 'l4:',  l4
  return ''

#print anagram(list1)


def anagram(l1):
  l2 = []
  l4 = []
  l5 = []
  for i in l1:
    #print i
    l3 = []
    for z in i:
      l3.append(z)
    l4.append( ''.join(l3[::-1] ) )
  for i in l1:
    for k in range(len(l4)):
      if i ==l4[k]:
        l5.append(i)
  #print 'here: ',l5
  return l5

#print anagram(list1)
'''
def bigram (l1):
  l2 = []
  l3 = []
  #l3 = l1[::-1]
  print l1
  for i in l1: 
    #print i[::-1]
    l3.append(i[::-1])
  print l3
  
  for i in range(len(l1)):
    #l2 = list(i)
    #l3 = l2[::-1]
    #print l2, l3
    #for z in l2:
    print l1[i], l3[i]
    if l1[i] == l3[i]:
      l2.append(l1[i])
  
  
  return l2

print 'hi: ',bigram(list1)
'''

def fib(a):
  l1 = [0, 1, 1, 2]
  b = a-len(l1)
  if a>4:
    while (b):
      b-=1
      print 'length: ',len(l1), sum(l1[-2:])
      l1.append(sum(l1[-2:]))
  else:
    l1 = l1[-b:]
  return a, len(l1), l1



#[0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
#0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...


#num = range(1,100)

def prime(num):
  z = [2]
  no = []
  for i in num:
    for k in range(2, i):
      #print i, k
      #print i%k
      print i, k, i%k
      if (i%k != 0):
        z.append(i)
      else:
        no.append(i)
  z = list(set(z))
  no = list(set(no))
  
  final = []
  

  for i in no:
    if (i in z):
      z.remove(i)
  

      #  print i, k
      #else:
      #  z.append(i)
   
  return z#.append()#, final# list(set(z) )

#print prime(num)


'''
def prime(list1):
  z = [1, 2, 3, 5, 7]
  for i in list1:
    if (i%2 != 0) and (i%3 !=0) and (i%5 !=0) and  (i%7 !=0) :
      #if (i):#**(1/2)==0):
      if (i%(i**0.5) == 0):#float(i*(1/2) )
        pass
      else:       
        z.append(i)

  print sorted(list(set(z)) )

prime(num)

'''











def my_func(x, y):
  z = []
  '''
  for i in x:
    for k in b:
      if i==k:
        z.append(i)
      else:
        pass
  return list(set(z))
  '''
  z = [a for a in x for b in y if a==b]
  return set(z)

