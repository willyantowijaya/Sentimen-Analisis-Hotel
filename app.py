import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import requests
import openpyxl
from time import sleep
from random import randint
import time
from datetime import datetime, timedelta
import nltk #import library nltk
from nltk.tokenize import word_tokenize #import word_tokenize for tokenizing text into words 
from nltk.corpus import stopwords #import stopwords
from nltk.tokenize import WordPunctTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory #import Indonesian Stemmer
import re #import regular expression
from stqdm import stqdm
import nltk
nltk.download('stopwords')

stop_words = (stopwords.words('indonesian'))
tok = WordPunctTokenizer()

emote = pd.read_csv('master_emoji.csv', encoding= 'unicode_escape')
emote_drop = ['ID', 'Sentiment', 'Makna Emoji', 'Special Tag']
emote = emote.drop(columns=emote_drop)
factory = StemmerFactory()
stemmer = factory.create_stemmer()
slang = pd.read_excel('kamus.xlsx')

def preprocessing(text):
    # casefolding
    text = text.lower()
    # stopwords removal
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # filtering emoji
    text = ' '.join([word for word in text.split() if word not in emote])
    # stemming
    text = stemmer.stem(text)
    text = (" ".join([x for x in tok.tokenize(text) if len(x) > 1])).strip()
    # normalisasi
    return ' '.join([slang[slang['before'] == word]['after'].values[0] if (slang["before"] == word).any() else word for word in text.split()])

df_list = pd.read_excel('Hotel_List.xlsx')
model = pickle.load(open('model1.pkl', 'rb'))

def main():
    st.title('Search Hotel Sentiment')
    nama_hotel = st.selectbox('Enter hotel name: ', df_list)
    if st.button('Search'):
        try:
            if nama_hotel:
                url_split = df_list[df_list['nama']==nama_hotel]['link'].values[0].split('-')
                pages = np.arange(1, 101, 5)
                Name = []
                Title = []
                Review = []
                Dates=[]

                for page in stqdm(pages,desc="Scraping"):
                    url = '-'.join([url_split[0], url_split[1], url_split[2], url_split[3], f'or{page}', url_split[4], url_split[5]])    # import the Url details to Python
                    headers["Connection"] = "keep-alive"
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'}
                    req = requests.get(url, headers=headers)
                    html = req.content
                    # Put it in soup
                    # sleep(randint(1, 2))
                    soup = BeautifulSoup(html, 'html.parser')
                    # for loop to extract Customer Names
                    for a in soup.find_all('a', {'class':'ui_header_link uyyBf'}):
                        Name.append(a.get_text(strip=True))
                    # for loop to extract Review Title
                    for a in soup.find_all('a', {'class':'Qwuub'}):
                        Title.append(a.get_text(strip=True))
                    # for loop to extract Reviews
                    for a in soup.find_all('q', {'class':'QewHA H4 _a'}):
                        Review.append(a.get_text())
                    for a in soup.findAll('div',{'class':'cRVSd'}):
                        Dates.append(a.span.text.strip()) 

                for i in range(len(Dates)):
                    d = Dates[i]
                    Dates[i]= d[-8:]

                df = pd.DataFrame({'Customer_name':Name,'Date': Dates, 'Review_Title':Title,'Review':Review})
                df = df.drop_duplicates(subset=['Review'])

                clean = []
                for i in stqdm(df.Review,desc="Preprocessing"):
                    clean.append(preprocessing(i))
                df['Preprocessing'] = clean
                df['Sentimens'] = model.predict(df.Preprocessing)
                data_sent = [len(df[df['Sentimens']=='Positive']), len(df[df['Sentimens']=='Negative'])]
                labels_sent = ['Positive: '+str(len(df[df['Sentimens']=='Positive']))+' data', f'Negative: '+str(len(df[df['Sentimens']=='Negative']))+'  data']
                colors = sns.color_palette('pastel')[0:2]
                fig1, ax = plt.subplots(1, 1, figsize=(6, 5))
                ax.pie(data_sent, labels = labels_sent, colors = colors, autopct='%.0f%%')
                centre_circle = plt.Circle((0,0),0.50,fc='white')
                ax.add_artist(centre_circle)
                st.pyplot(fig1)
                pos_tot = []
                neg_tot = []
                for i, j in df.groupby('Date'):
                    pos_tot.append(len(j[j['Sentimens']=='Positive']))
                    neg_tot.append(len(j[j['Sentimens']=='Negative']))
                fig2, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df.Date.unique(), pos_tot, label='Positive', color='blue')
                ax.plot(df.Date.unique(), neg_tot, label='Negative', color='red')
                ax.set_ylabel('Total Sentiment')
                ax.invert_xaxis()
                ax.legend()
                plt.setp(ax.get_xticklabels(), rotation=70, horizontalalignment='right')
                st.pyplot(fig2)
                df = df.drop(columns=['Preprocessing'])
                st.dataframe(df)
                st.download_button('Download Data', df.to_csv(), file_name='Hasil Sentimen')
        except:
            st.warning('Hotel has no review or try again')


if __name__ == '__main__':
    main()