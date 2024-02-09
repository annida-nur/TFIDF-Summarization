
##########################################################################
#libraries & packages
import streamlit as st
import PyPDF2

import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import os
##########################################################################
stopword_in =  set(stopwords.words('indonesian'))
stopword_en =  set(stopwords.words('english'))
stopword_both = stopword_in.union(stopword_en)

factory = StemmerFactory()
stemmer_in = factory.create_stemmer()
stemmer_en = PorterStemmer()
##########################################################################
# sidebar
st.sidebar.image(
    "https://miro.medium.com/v2/resize:fit:720/format:webp/1*bdumEoU0MlHHEpiF5gn81w.jpeg",
    width = 300
)
st.sidebar.title("Hi there, Welcome 👋")
st.sidebar.caption("""
            Want to know the main content quickly, but lazy to read a long text? Summarizing is the answer! 
            Summarize your text here quickly, set the number of sentences in the summary yourself. 
            We'll be happy to help you. Hope you enjoy your time here 😄.
            """)
page = st.sidebar.selectbox("Menu",("Summarize Direct Text","Summarize Text Files"))
st.sidebar.caption("Creator : Annida Nur Islami [(LinkedIn)](https://www.linkedin.com/in/annida-nur-islami-a23694214/)")
##########################################################################
#page1
if page == "Summarize Direct Text":
    st.title(f"{page} Menu")
    text = st.text_area('Write down the text here : ',height=200)
    compress = st.slider('Compression (%) : ', 0, 100, 25)
    ok = st.button("Summarize")
    
    if ok :
        # split sentence
        text_str = text.replace('\n', '')
        sentences = re.split('\. |\.',text_str)
        # hapus tanda baca, angka, dan karakter khusus
        clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
        # membuat huruf kecil
        clean_sentences = [s.lower() for s in clean_sentences]
        
        #memfilter kata/token penting (seluruh kata kecuali yang termasuk stopword)
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        tokenized = [tokenizer.tokenize(s.lower()) for s in clean_sentences]
        important_token = []
        for sent in tokenized:
            filtered = [s for s in sent if s not in stopword_both]
            important_token.append(filtered)
            
        #menggabungkan kata di list yang telah terfilter menjadi sebuah kalimat
        sw_removed = [' '.join(t) for t in important_token]
        
        #mengubah kata menjadi kata dasarnya
        stemmed_sent = [stemmer_in.stem(sent) for sent in sw_removed]
        stemmed_sent = [stemmer_en.stem(sent) for sent in sw_removed]
        #ektraksi fitur (tfidf)
        vec = TfidfVectorizer(lowercase=True)
        document = vec.fit_transform(stemmed_sent)
        document = document.toarray()
        
        # n adalah variabel integer panjang hasil ringkasan
        n = int((compress/100)*len(sentences))
        #result merupakan variabel yang menyimpan bobot setiap kalimat
        #seberapa penting kalimat tersebut terhadap keseluruhan teks
        result = np.sum(document, axis=1)
        #diurutkan
        result = sorted(result)
        #diambil index
        top_n = np.argsort(result)[-n:]
        summ_index = sorted(top_n)
        
        result = []
        for i in summ_index:
            x = sentences[i] + '. \n'
            result.append(x)
        result = "".join(result)
    
        st.header("Summary Result")
        st.caption(f"({len(sentences)} Sentences ➡ {n} Sentences)")
        st.write(result)
##########################################################################
#page2  
else :
    st.title(f"{page} Menu")
    upload_file = st.file_uploader("Upload Your Text File", type = ['txt'])
    compress = st.slider('Compression (%) : ', 0, 100, 25)
    ok = st.button("Summarize")
    
    if ok:
        if upload_file:
            file_name = upload_file.name
            file_name = file_name.split(".",1)
            file_extension = file_name[1]
            
            sentences = []
            for line in upload_file:
                line = line.decode()
                sentences.append(line)
            text = ' '.join(map(str, sentences))
            st.subheader("Text of The File 📜")
            with st.expander(upload_file.name, expanded=False):
                st.write(text)

            # split sentence
            text_str = text.replace('\n', '')
            sentences = re.split('\. |\.',text_str)
            # hapus tanda baca, angka, dan karakter khusus
            clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
            # membuat huruf kecil
            clean_sentences = [s.lower() for s in clean_sentences]

            #memfilter kata/token penting (seluruh kata kecuali yang termasuk stopword)
            tokenizer = nltk.RegexpTokenizer(r"\w+")
            tokenized = [tokenizer.tokenize(s.lower()) for s in clean_sentences]
            important_token = []
            for sent in tokenized:
                filtered = [s for s in sent if s not in stopword_both]
                important_token.append(filtered)

            #menggabungkan kata di list yang telah terfilter menjadi sebuah kalimat
            sw_removed = [' '.join(t) for t in important_token]

            #mengubah kata menjadi kata dasarnya
            stemmed_sent = [stemmer_in.stem(sent) for sent in sw_removed]
            stemmed_sent = [stemmer_en.stem(sent) for sent in sw_removed]
            #ektraksi fitur (tfidf)
            vec = TfidfVectorizer(lowercase=True)
            document = vec.fit_transform(stemmed_sent)
            document = document.toarray()

            # n adalah variabel integer panjang hasil ringkasan
            n = int((compress/100)*len(sentences))
            #result merupakan variabel yang menyimpan bobot setiap kalimat
            #seberapa penting kalimat tersebut terhadap keseluruhan teks
            result = np.sum(document, axis=1)
            #diurutkan
            result = sorted(result)
            #diambil index
            top_n = np.argsort(result)[-n:]
            summ_index = sorted(top_n)

            result = []
            for i in summ_index:
                x = sentences[i] + '. \n'
                result.append(x)
            result = "".join(result)
            st.header("Summary Result")
            st.caption(f"({len(sentences)} Sentences ➡ {n} Sentences)")
            st.write(result)
            
            st.download_button(
                'Download Result', 
                result, 
                "result.txt",
                "text/plain",
                key='download-text'
            )
