"""# **Preprocessing**
""

# import pandas as pd
# import numpy as np
#
# test_data = pd.read_csv('hotel_reviewss.csv')
#
# test_data
#
# """#Library Preprocessing"""
#
# import Sastrawi
# import nltk
# import pandas as pd
# nltk.download('punkt')
#
# import nltk #import library nltk
# from nltk.tokenize import word_tokenize #import word_tokenize for tokenizing text into words
# from nltk.tokenize import sent_tokenize #import sent_tokenize for tokenizing paragraph into sentences
# from nltk.stem.porter import PorterStemmer #import Porter Stemmer Algorithm
# from nltk.stem import WordNetLemmatizer #import WordNet lemmatizer
# from nltk.corpus import stopwords #import stopwords
# from nltk.tokenize import WordPunctTokenizer
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory #import Indonesian Stemmer
# from bs4 import BeautifulSoup
# import re #import regular expression
#
# """Case Folding"""
#
# t = test_data['Review'].str.lower()
#
# print('Sebelum Case Folding : \n')
# print(test_data['Review'].head(5))
#
# print('Sesudah Case Folding : \n')
# print(t.head(5))
#
# """Tokenizing, Filtering, Stemming"""
#
# emote = pd.read_csv('master_emoji.csv', encoding= 'unicode_escape')
# emote_drop = ['ID', 'Sentiment', 'Makna Emoji', 'Special Tag']
# emote = emote.drop(columns=emote_drop)
#
# factory = StemmerFactory()
# stemmer = factory.create_stemmer()
#
# nltk.download('stopwords')
# stop_words = (stopwords.words('indonesian'))
# tok = WordPunctTokenizer()
#
#
# def proses_teks(teks):
#     soup = BeautifulSoup(teks, 'lxml')
#     souped = soup.get_text()
#     try:
#         teks = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
#     except:
#         teks = souped
#     teks_bersih = ' '.join([word for word in teks.split() if word not in stop_words])
#     teks_bersih = ' '.join([word for word in teks_bersih.split() if word not in emote])
#     teks_bersih = stemmer.stem(teks_bersih)
#     return (" ".join([x for x in tok.tokenize(teks_bersih) if len(x) > 1])).strip()
#
# data=[]
# for x in test_data.Review:
#   data.append(proses_teks(x))
#   print(len(data),'/',len(test_data))
# data
#
# test_data.Review
#
# """Normalisasi"""
#
# import pandas as pd
# df = test_data
# slang = pd.read_excel('kamus.xlsx')
# df['normal'] = data
# for idx, row in slang.iterrows():
#     df['normal'] = df.normal.replace(r"\b"+row['before']+r"\b", row['after'], regex=True)
#
# df = df.drop_duplicates(subset=['normal'])
#
# df
#
# """Save Preprocessing"""
#
# df.to_csv("text_preprocessing.csv")

"""# **SVM**"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# df = test_data

df = pd.read_csv('text_preprocessing.csv')

"""SVM Data Prepocessing"""

X = df['normal']
y = df['Sentimens']

df[df.Sentimens=='Negative']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

"""Training Data"""

y_train

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
tf= TfidfVectorizer()
svc = SVC(kernel='linear')
model= Pipeline([('vectorizer', tf), ('classifier', svc)])
model.fit(X_train, y_train)
result= model.predict(X_test)

import pickle

pickle.dump(model, open('model.pkl', 'wb'))

"""Predictions"""

y_pred = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

from sklearn.metrics import classification_report,confusion_matrix
report=classification_report(y_test, result)
print(report)

import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
group_names = ['True Neg','False Neg','False Pos','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}" for v1, v2 in
zip(group_names,group_counts)]
labels = np.asarray(labels).reshape(2,2)
ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Negative','Positive'])
ax.yaxis.set_ticklabels(['Negative','Positive'])
## Display the visualization of the Confusion Matrix.
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('text_preprocessing.csv')

df['Sentimens']=model.predict(df.normal)

# nama_hotel = 'Sofitel'
# df_result = df[df['Review'].str.contains(nama_hotel.lower())].append(df[df['Review'].str.contains(nama_hotel)])
# df_result

data_neg = [len(df[df['Sentimens']=='Negative']), len(df)-len(df[df['Sentimens']=='Negative'])]
data_pos = [len(df[df['Sentimens']=='Positive']), len(df)-len(df[df['Sentimens']=='Positive'])]
labels_neg = ['Negative', ' ']
labels_pos = ['Positive', ' ']
colors = sns.color_palette('pastel')[0:2]

plt.pie(data_neg, labels = labels_neg, colors = colors, autopct='%.0f%%')
centre_circle = plt.Circle((0,0),0.50,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.show()

plt.pie(data_pos, labels = labels_pos, colors = colors, autopct='%.0f%%')
centre_circle = plt.Circle((0,0),0.50,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.show()

pos_tot = []
neg_tot = []
for i, j in df.groupby('Date'):
    pos_tot.append(len(j[j['Sentimens']=='Positive']))
    neg_tot.append(len(j[j['Sentimens']=='Negative']))
plt.figure(figsize=(19, 10))
plt.plot(df.Date.unique(), pos_tot, label='Positive', color='blue')
plt.plot(df.Date.unique(), neg_tot, label='Negative', color='red')
plt.legend()
plt.xticks(rotation=90)
plt.show()

df['Date'].apply(lambda x: x.split(' ')[0])

df['Date'].head()

from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
from time import sleep
from random import randint

df

np.arange(1, 601, 5)

# https://www.tripadvisor.co.id/Hotel_Review-g297698-d5039960-Reviews-Sofitel_Bali_Nusa_Dua_Beach_Resort-Nusa_Dua_Nusa_Dua_Peninsula_Bali.html

# df_list = pd.DataFrame({'nama':['sofitel'],
#               'link':['https://www.tripadvisor.co.id/Hotel_Review-g297698-d5039960-Reviews-Sofitel_Bali_Nusa_Dua_Beach_Resort-Nusa_Dua_Nusa_Dua_Peninsula_Bali.html']})

nama_hotel = 'sofitel'

df_list = pd.DataFrame({'nama':['sofitel'],
              'link':['https://www.tripadvisor.co.id/Hotel_Review-g297698-d5039960-Reviews-Sofitel_Bali_Nusa_Dua_Beach_Resort-Nusa_Dua_Nusa_Dua_Peninsula_Bali.html']})

model = pickle.load(open('model.pkl', 'rb'))
url_split = df_list[df_list['nama']==nama_hotel]['link'][0].split('-')

pages = np.arange(1, 101, 5)
Name = []
Title = []
Review = []
Dates=[]

for page in pages:
    url = '-'.join([url_split[0], url_split[1], url_split[2], url_split[3], f'or{page}', url_split[4], url_split[5]])    # import the Url details to Python
    req = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = req.content
    # Put it in soup
    sleep(randint(1, 2))  
    soup = BeautifulSoup(html, 'html.parser')
    # for loop to extract Customer Names
    for a in soup.find_all('a', {'class':'ui_header_link bPvDb'}):
        Name.append(a.get_text(strip=True))
    # for loop to extract Review Title
    for a in soup.find_all('a', {'class':'fCitC'}):
        Title.append(a.get_text(strip=True))
    # for loop to extract Reviews
    for a in soup.find_all('q', {'class':'XllAv H4 _a'}):
        Review.append(a.get_text())
    for a in soup.findAll('div',{'class':'bcaHz'}):
        Dates.append(a.span.text.strip()) 
        
for i in range(len(Dates)):
    d = Dates[i]
    Dates[i]= d[-8:]

df = pd.DataFrame({'Customer_name':Name,'Date': Dates, 'Review_Title':Title,'Review':Review})

df['Sentimens'] = model.predict(df.normal)
(df)

data_neg = [len(df[df['Sentimens']=='Negative']), len(df)-len(df[df['Sentimens']=='Negative'])]
data_pos = [len(df[df['Sentimens']=='Positive']), len(df)-len(df[df['Sentimens']=='Positive'])]
labels_neg = ['Negative', ' ']
labels_pos = ['Positive', ' ']
colors = sns.color_palette('pastel')[0:2]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].pie(data_pos, labels = labels_pos, colors = colors, autopct='%.0f%%')
centre_circle = plt.Circle((0,0),0.50,fc='white')
ax[0].add_artist(centre_circle)
ax[1].pie(data_neg, labels = labels_neg, colors = colors, autopct='%.0f%%')
centre_circle = plt.Circle((0,0),0.50,fc='white')
ax[1].add_artist(centre_circle)

pos_tot = []
neg_tot = []
for i, j in df.groupby('Date'):
    pos_tot.append(len(j[j['Sentimens']=='Positive']))
    neg_tot.append(len(j[j['Sentimens']=='Negative']))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.Date.unique(), pos_tot, label='Positive', color='blue')
ax.plot(df.Date.unique(), neg_tot, label='Negative', color='red')
ax.legend()
plt.setp(ax.get_xticklabels(), rotation=70, horizontalalignment='right')
plt.show()

data_neg = [len(df[df['Sentimens']=='Negative']), len(df)-len(df[df['Sentimens']=='Negative'])]
data_pos = [len(df[df['Sentimens']=='Positive']), len(df)-len(df[df['Sentimens']=='Positive'])]
labels_neg = ['Negative', ' ']
labels_pos = ['Positive', ' ']
colors = sns.color_palette('pastel')[0:2]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].pie(data_pos, labels = labels_pos, colors = colors, autopct='%.0f%%')
centre_circle = plt.Circle((0,0),0.50,fc='white')
ax[0].add_artist(centre_circle)
ax[1].pie(data_neg, labels = labels_neg, colors = colors, autopct='%.0f%%')
centre_circle = plt.Circle((0,0),0.50,fc='white')
ax[1].add_artist(centre_circle)

pos_tot = []
neg_tot = []
for i, j in df.groupby('Date'):
    pos_tot.append(len(j[j['Sentimens']=='Positive']))
    neg_tot.append(len(j[j['Sentimens']=='Negative']))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.Date.unique(), pos_tot, label='Positive', color='blue')
ax.plot(df.Date.unique(), neg_tot, label='Negative', color='red')
ax.legend()
plt.setp(ax.get_xticklabels(), rotation=70, horizontalalignment='right')
plt.show()



df = pd.DataFrame({'Customer_name':Name,'Date': Dates, 'Review_Title':Title,'Review':Review})
df



import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import requests
from time import sleep
from random import randint

df_list = pd.read_excel('list_hotel.xlsx')
model = pickle.load(open('model.pkl', 'rb'))

nama_hotel = 'melia bali'

if nama_hotel:
    url_split = df_list[df_list['nama']==nama_hotel]['link'][0].split('-')
    pages = np.arange(1, 101, 5)
    Name = []
    Title = []
    Review = []
    Dates=[]

    for page in pages:
        url = '-'.join([url_split[0], url_split[1], url_split[2], url_split[3], f'or{page}', url_split[4], url_split[5]])    # import the Url details to Python
        req = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = req.content
        # Put it in soup
        sleep(randint(1, 2))  
        soup = BeautifulSoup(html, 'html.parser')
        # for loop to extract Customer Names
        for a in soup.find_all('a', {'class': 'ui_header_link bPvDb'}):
            Name.append(a.get_text(strip=True))
        # for loop to extract Review Title
        for a in soup.find_all('a', {'class': 'fCitC'}):
            Title.append(a.get_text(strip=True))
        # for loop to extract Reviews
        for a in soup.find_all('q', {'class': 'XllAv H4 _a'}):
            Review.append(a.get_text())
        # for loop to extract Dates
        for a in soup.findAll('div', {'class': 'bcaHz'}):
            Dates.append(a.span.text.strip())

    for i in range(len(Dates)):
        d = Dates[i]
        Dates[i]= d[-8:]

    df = pd.DataFrame({'Customer_name':Name,'Date': Dates, 'Review_Title':Title,'Review':Review})

    df['Sentimens'] = model.predict(df.normal)

    data_neg = [len(df[df['Sentimens']=='Negative']), len(df)-len(df[df['Sentimens']=='Negative'])]
    data_pos = [len(df[df['Sentimens']=='Positive']), len(df)-len(df[df['Sentimens']=='Positive'])]
    labels_neg = ['Negative', ' ']
    labels_pos = ['Positive', ' ']
    colors = sns.color_palette('pastel')[0:2]
    fig1, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].pie(data_pos, labels = labels_pos, colors = 'blue', autopct='%.0f%%')
    centre_circle = plt.Circle((0,0),0.50,fc='white')
    ax[0].add_artist(centre_circle)
    ax[1].pie(data_neg, labels = labels_neg, colors = 'red', autopct='%.0f%%')
    centre_circle = plt.Circle((0,0),0.50,fc='white')
    ax[1].add_artist(centre_circle)
#     st.pyplot(fig1)

    pos_tot = []
    neg_tot = []
    for i, j in df.groupby('Date'):
        pos_tot.append(len(j[j['Sentimens']=='Positive']))
        neg_tot.append(len(j[j['Sentimens']=='Negative']))

    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.Date.unique(), pos_tot, label='Positive', color='blue')
    ax.plot(df.Date.unique(), neg_tot, label='Negative', color='red')
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=70, horizontalalignment='right')
#     st.pyplot(fig2)
#     st.dataframe(df)

nama_hotel = 'sofitel'
df_list[df_list['nama'] == nama_hotel]['link'].values[0].split('-')

