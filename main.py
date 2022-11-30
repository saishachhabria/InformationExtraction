from utils import *
from build_components import *
from knowledge_graph import *

if __name__ == '__main__':
    documents, file_list = read_files()     # Unstructured data
    documents = clean_data(documents)       # Data Cleaning
    build_components(documents, file_list)  # Entity Extraction, Named Entity Recognition and Coreference Resolution
    extract_relations()                     # Relationship Extraction
    build_knowledge_graph()                 # Post Processing and Knowledge Graph
