# pandas, sklearn => 
# numpy

import pandas as pd
import numpy as np

from time import time
from time import sleep
from random import randint
from IPython.core.display import clear_output
from warnings import warn
from bs4 import BeautifulSoup
import requests




'''
stop_words = set(stopwords.words('english')) 


print(word_tokenize(data))# ['All', 'work', 'and', 'no', 'play']
print (sent_tokenize(sentences)) #['sent1', 'sent2']

sent_root = []

for i in df['Journal title']:
 
  word_tokens = (word_tokenize(str(i)))
  filtered_sentence = [w for w in word_tokens if not w in stop_words] 
  #print filtered_sentence
  sent = []
  for token in filtered_sentence:
     try: 
         k = (stemmer.stem(token))
         sent.append(k)
     except:
         sent.append(token)
         pass

  sent_root.append(sent)

'''
#pd.DataFrame(data, columns = ['Name', 'Age'])  
 
'''
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!')

'''
 
#word_tokens = word_tokenize(example_sent) 
  

  # Assignment
# Write a program to scrape news from the https://www.space.com/news web page.
# Submit a notebook that contains a function that retrieves the space.com news page,
# extracts news stories, and prints the headline, author, synopsis, date and time for each story.

# 'date_time' = <time class="published-date relative-date" datetime="2019-11-30T13:12:52Z" data-published-date="2019-11-30T13:12:52Z">13 hours ago</time>

# Use Chrome Developer tools to examine the structure of the page to find the HTML for the articles,
# then use requests and BeautifulSoup to extract the data that you need.

# Some of the items in the news list are sponsored ads; and even though they look like news stories,
# they are structurally different in the code. You can ignore those and just scrape the news itself.

# news stories are identified by: <div class="listingResult small result1 " data-page="1">
# ads are identified by this code: <div class="listingResult small result2 sponsored-post">

# If you wish to challenge yourself, have your spider follow the next link at the bottom of the page and scrape more news stories.

# make an initial call
#url = 'https://api.harvardartmuseums.org/person?apikey=a3e57b20-0fee-11ea-8207-4d8b2f0539bd&q=culture:British'
#page = 0
#query = {'fields': 'displayname,birthplace,culture', 'page': page}
#response = requests.get(url, query)
#print('Querying {}'.format(response.url))

#API in python

#import requests
#url = 'https://api.harvardartmuseums.org/Person?apikey=8e309350-108e-11ea-bc83-71d2fa26d53e'
#response = requests.get(url)
#print respone

#query={'limit':99}
#response = requests.get(url,query)
#response.ok
#response.json()["info"]
#create list of dictionaries instead of list of json object


# define a function that will process the data for us
#def process_data(birth_place):
#iterate the results and only grab the properties that we are interested in
#return [{ 'birthplace':birth['birthplace'], 'displayname': birth['displayname'], 'culture': birth['culture'] } for birth in birth_place]
# declare a list to store all results
births = []
page = 0
# declare variables to track skip amount
#process_data(births)

# declare variables to track skip amount
'''
# make an initial call

url = 'https://api.harvardartmuseums.org/person?apikey=64a386b0-7a8c-11ea-b6eb-a3d44d57a103'
#I could not format the apikey into the quary. Is this typical or did I format it incorrectly?
#play with api in quary
query = {
    'q' : 'culture : British',
    'size' : '100',
    'page' : '1'
}
response = requests.get(url,query)
data = response.json()
print (data)


url = 'https://api.harvardartmuseums.org/person?apikey=a3e57b20-0fee-11ea-8207-4d8b2f0539bd&q=culture:British'
response = requests.get(url)
print('Querying {}'.format(response))


data = response.json()

names = []
birthplace = []
for i in range( len(data['records'])):
  names.append( data['records'][i]['displayname'] )
  birthplace.append( data['records'][i]['birthplace'] )
print (names, birthplace)
# make sure we got a valid response
'''
'''
if(response.ok):
  # get the full data from the response
  data = response.json()
  # get the meta data
  inform = data['info']

  total = inform['pages']
  print('There is a total of {} pages to fetch'.format(total))


page = 0

while page < total:
  print(page)
  page = page + 1

  query = {'page':page}
  response = requests.get(url, params=query)
  print('Querying{}'.format(response.url))
  if (response.ok):
    # now incidents will be old values + new values
    print('{} results processed so far'.format(len(incidents)))
    #increment page
    #page = page + 1
'''


url = 'https://www.space.com/news'
response = requests.get(url)
data = response.text
soup = BeautifulSoup(data, 'html.parser')




'''
dates = []
for i in soup.findAll('time'):
        if i.has_attr('datetime'):
            dates.append(i['datetime'])
            print(i['datetime'])

print dates

'''
# raw_articles = soup.select('.listingResult*')


raw_articles = soup.find_all('div', {'class': ['listingResult', 'small', '*']})

#print soup.find_all('div')
# print(type(raw_articles))

articles = []

#print raw_articles

for article in raw_articles:

    try:
        #headline = article.select_one('h3').get_text()
        #print headline
        author = article.select_one('header > p > span > span').get_text()
        print author
        synopsis = article.select_one('.content > p').get_text()
        print (synopsis)
        #print ('done')
        #datetime = article.select_one('time').get_text()
        #datetime = article.find('header > p > time').get_text()
       
        #article = {'headline': headline, 'author': author, 'synopsis': synopsis} # , 'datetime': datetime
        #articles.append(article)
        #print datetime
        #print author.replace('\n', '')
        #, type(synopsis)
    except:
        print('This exception exists because sponsored ads do not have h3 tags.')




from bs4 import BeautifulSoup
import requests

url = 'https://www.space.com/news'
response=requests.get(url)

data=response.text
soup=BeautifulSoup(data,'html.parser')


def process_page(soup):
  raw_space_news=soup.select('.content')
  if(response.ok): 
    new_stories = [] 
    spacestories=[]
    for spacenews in raw_space_news:
      data=response.text
      headline=soup.select_one('.article-name').get_text().strip()
      author=soup.select_one('.by-author > span').get_text().strip()
      synopsis=soup.select_one('.synopsis').get_text().strip() 
      date_time=soup.select_one('time').get('datetime')
      new_story = {'headline': headline, 'author': author, 'synopsis': synopsis,'date_time':date_time}
      spacestories.append(new_story)
    return spacestories
  
  

print process_page(soup)





def user_contacts(users):
  dictionary = {}
  
  l1 = []
  l2 = []
  
  for i in users:
    try:
      l1.append(i[0])
      l2.append(i[1])
    except:
      l2.append(None)

  #print l1, l2    
  
#user_contacts(l1)
'''
  for x in users:
    if len(x) == 2:
      x[0] = dictionary.keys()
      x[1] = dictionary.values()
    #print(dictionary)
user_contacts(l1)
'''


'''
bos = load_boston()
diab =load_diabetes()

print (bos.data.shape)
print (diab.data.shape)

#X_b, y_b = bos.data, bos.target
X_d, y_d = diab.data, diab.target

scale = StandardScaler()
X_d = scale.fit_transform(X_d)

y_d = pd.DataFrame(y_d)
y_d = y_d.astype("category")

x_train, x_val, y_train, y_val = train_test_split(X_d, y_d.values, test_size = 0.25)

clf = linear_model.LogisticRegression()

clf.fit(x_train, y_train)

test = clf.predict(x_val)

test = test.reshape(-1,1)
#y_val = y_val.flatten()

#print test.shape, y_val.shape
#print (cross_val_score(clf, test, y_val, cv = 3, scoring= "f1_macro").mean() )


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
X = [['Male', 1], ['Female', 3], ['Female', 2]]
X = enc.fit_transform(X)
print X
'''

'''
df_b = pd.DataFrame(bos.data)
#df_d = pd.DataFrame(diab.data)
b_y = pd.DataFrame(bos.target)
#d_y = pd.DataFrame(diab.target)
# from numeric to categorical
#d_y = d_y.astype('category')
#print d_y.dtypes
# from cat to numeric
print "here \n {}".format(b_y.head() )


#clf = linear_model.LogisticRegression()
#model1 = clf.fit(X_d, d_y.values)

clf2 = linear_model.Ridge()
model2 = clf2.fit(df_b.values, b_y.values)
#model2 = ridge_regression.fit(df_d, df_d)

new = model2.predict(X_b)
#new2 = model1.predict(X_d)
#print new2[:5]
#print new
'''

