import os

try: import nltk
except: os.system("python3.7 -m pip install nltk")

try: import pandas
except: os.system("python3.7 -m pip install pandas")

try: import glob
except: os.system("python3.7 -m pip install glob")

try: import spacy
except: os.system("python3.7 -m pip install spacy==2.1.0")

try: import stanfordcorenlp
except: os.system("python3.7 -m pip install stanfordcorenlp")

try: import neuralcoref
except: os.system("python3.7 -m pip install neuralcoref")

try: import en_core_web_lg
except: os.system("python3.7 -m spacy download en_core_web_lg")

os.system('sudo apt install openjdk-8-jre-headless')