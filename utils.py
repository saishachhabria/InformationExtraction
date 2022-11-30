# Imports

import os
import glob

import re
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')

path = os.getcwd() + '/'

# Data Loading

def read_files():
    """
    Reads data from files in the data/input directory
    """
    
    file_list = []
    for f in glob.glob(path + 'data/input/*'):
        file_list.append(f)

    documents = []

    for file in file_list:    
        with open(file,"r") as f:
            lines = f.read().splitlines()
        doc = ""
        for line in lines:
            doc += line
        documents.append(doc)
    
    return (documents, file_list)

# Data Cleaning

def remove_stop_words(text):
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def normalize(text):
    return text.lower()

def remove_acronyms(text):
    text = re.sub(r"\b[A-Z\.]{2,}\b", "", text) 
    return text

def clear_unicode(text):
    text_encode = text.encode(encoding="ascii", errors="ignore")
    # decoding the text
    text_decode = text_encode.decode()
    # cleaning the text to remove extra whitespace 
    clean_text = " ".join([word for word in text_decode.split()])
    return clean_text

def remove_special_chars(text):
    text = re.sub("@\S+", "", text) # Removes mentions @
    text = re.sub("https?:\/\/.*[\r\n]*", "", text) # Removes URLs
    text = re.sub("#", "", text) # Removes hashtags
    text = re.sub("Mr", "", text)
    text = re.sub("Mrs", "", text)
    text = re.sub("Ms", "", text)
    text = re.sub("MFP", "", text) # Removes hashtags
    return text

def remove_punctuations(text):
    punct = set(string.punctuation) 
    text = "".join([ch for ch in text if ch not in punct])
    return text

def stemming(text):
    stemmer = PorterStemmer()
    text_stemmed = [stemmer.stem(y) for y in text.split(' ')]
    text = ' '.join(text_stemmed)
    return text

def clean_data(documents):
    """
    Performs predefined set of cleaning operations on the data
    """
    
    for i in range(len(documents)):
        # documents[i] = remove_stop_words(documents[i])
        # documents[i] = stemming(documents[i])
        documents[i] = clear_unicode(documents[i])
        documents[i] = remove_special_chars(documents[i])
        documents[i] = remove_acronyms(documents[i])

    return documents
        