import os
import nltk
import pickle

# For NER
import spacy
from spacy import displacy
from collections import Counter
# import en_core_web_trf
import en_core_web_lg
from pprint import pprint

# For coreference resolution
import json
from stanfordcorenlp import StanfordCoreNLP
import neuralcoref
import gc

gc.collect()
path = os.getcwd() + '/'

# Named Entity Recognition

class StanfordNER:
    
    def __init__(self):
        self.get_stanford_ner_location()

    def get_stanford_ner_location(self):
        loc = path + "stanford-ner-2018-10-16"
        self.stanford_ner_tagger = nltk.tag.StanfordNERTagger(loc+'/classifiers/english.all.3class.distsim.crf.ser.gz', loc + '/stanford-ner.jar')

    def ner(self, doc):
        sentences = nltk.sent_tokenize(doc)
        result = []
        for sent in sentences:
            words = nltk.word_tokenize(sent)
            tagged = self.stanford_ner_tagger.tag(words)
            result.append(tagged)
        return result
    
    def ner_to_dict(self,ner):
        """
        Expects ner of the form list of tuples 
        """
        ner_dict = {}
        for tup in ner:
            ner_dict[tup[0]] = tup[1]
        return ner_dict

    def display(self, ner):
        print(ner)
        print("\n")


class SpacyNER:
    
    def ner(self, doc):    
        nlp = en_core_web_lg.load()
        # nlp = en_core_web_sm.load()
        doc = nlp(doc)
        return [(X.text, X.label_) for X in doc.ents]
    
    def ner_to_dict(self, ner):
        """
        Expects ner of the form list of tuples 
        """
        ner_dict = {}
        for tup in ner:
            ner_dict[tup[0]] = tup[1]
        return ner_dict
    
    def display(self, ner):
        print(ner, end = '\n\n')


class NltkNER:
    
    def ner(self, doc):
        pos_tagged = self.assign_pos_tags(doc)
        #chunks = self.split_into_chunks(pos_tagged)
        result = []
        for sent in pos_tagged:
            result.append(nltk.ne_chunk(sent))
        return result

    def assign_pos_tags(self, doc):
        sentences = nltk.sent_tokenize(doc)
        words = [nltk.word_tokenize(sent) for sent in sentences]
        pos_tagged = [nltk.pos_tag(word) for word in words]
        return pos_tagged
    
    def split_into_chunks(self, sentences):
        # This rule says that an NP chunk should be formed whenever the chunker finds an optional determiner (DT) or possessive pronoun (PRP$) followed by any number of adjectives (JJ/JJR/JJS) and then any number of nouns (NN/NNS/NNP/NNPS) {dictator/NN Kim/NNP Jong/NNP Un/NNP}. Using this grammar, we create a chunk parser.
        grammar = "NP: {<DT|PRP\$>?<JJ.*>*<NN.*>+}"
        cp = nltk.RegexpParser(grammar)
        chunks = []
        for sent in sentences:
            chunks.append(cp.parse(sent))
        return chunks

    def display(self,ner):
        print("\n\nTagged: \n\n")
        pprint(ner)
        print("\n\nTree: \n\n ")
        for leaves in ner:
            print(leaves)
            #leaves.draw()
        print("\n")


# Coreference Resolution (Using neuralcoref by spaCy)

def resolve_coreferences_neural(doc):
    nlp = en_core_web_lg.load()
    neuralcoref.add_to_pipe(nlp)
    processed_text = nlp(doc)
    resolved_text = processed_text._.coref_resolved
    return resolved_text


def build_components(documents, file_list):
    
    options = ['spacy']

    output_path = path + "data/output/"
    ner_pickles_op = output_path + "ner/"
    coref_cache_path = output_path + "caches/"
    coref_resolved_op = output_path + "kg/"
    stanford_core_nlp_path = path + "stanford-corenlp-4.5.1/"

    os.environ['STANFORD_PARSER'] = stanford_core_nlp_path
    os.environ['STANFORD_MODELS'] = stanford_core_nlp_path

    spacy_ner = SpacyNER() # Remove this later on

    named_entities = None

    print('Building components - Entity extraction, named entity recognition and coreference resolution\n')
    for j in range(len(documents)):

        doc = documents[j]

        print('\nWorking on file', file_list[j].split('/')[-1])
        print("Using Spacy for NER & Resolving Coreferences")

        # doc = resolve_coreferences_neural(doc)

        for i in range(0, len(options)):
            if(options[i] == "nltk"):
                print("Using NLTK for NER")
                nltk_ner = NltkNER()
                named_entities = nltk_ner.ner(doc)
                nltk_ner.display(named_entities)
                # ToDo -- Implement ner_to_dict for nltk_ner
                spacy_ner = SpacyNER()
                named_entities = spacy_ner.ner_to_dict(spacy_ner.ner(doc))
            elif(options[i] == "stanford"):
                print("Using Stanford for NER")
                stanford_ner = StanfordNER()
                tagged = stanford_ner.ner(doc)
                ner = stanford_ner.ner(doc)
                stanford_ner.display(ner)
                # ToDo -- Implement ner_to_dict for stanford_ner
                named_entities = spacy_ner.ner_to_dict(spacy_ner.ner(doc))
            elif(options[i] == "spacy"):
                spacy_ner = SpacyNER()
                named_entities = spacy_ner.ner(doc)
                spacy_ner.display(named_entities)
                named_entities = spacy_ner.ner_to_dict(named_entities)
        
        # Save named entities
        op_pickle_filename = ner_pickles_op + "named_entity_" + file_list[j].split('/')[-1].split('.')[0] + ".pickle"
        with open(op_pickle_filename,"wb") as f:
            pickle.dump(named_entities, f)

        op_filename = coref_resolved_op + file_list[j].split('/')[-1]
        with open(op_filename,"w+") as f:
            f.write(doc)

        print('Output pickle file -->', op_pickle_filename)
        print('Output text file -->', op_filename)
    print('\n--------------------------------------------------------------------------------\n')
    