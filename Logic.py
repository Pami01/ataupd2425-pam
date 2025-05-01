import json
import nltk 
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn_crfsuite
import pickle
from sklearn_crfsuite import metrics
from  collections import Counter
import matplotlib.pyplot as plt
import sklearn.metrics
import sklearn_crfsuite.metrics 
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
from nltk.corpus import wordnet as wn

entity_labels = ["O","anatomical location","animal","biomedical technique","bacteria","chemical","dietary supplement","DDF","drug","food","gene","human","microbiome","statistical technique"]
locations = ["title","abstract"]
documents_dev = [r"data\Annotations\Dev\json_format\dev.json"]
documents_train = [r"data\Annotations\Train\bronze_quality\json_format\train_bronze.json",r"data\Annotations\Train\gold_quality\json_format\train_gold.json",r"data\Annotations\Train\platinum_quality\json_format\train_platinum.json",r"data\Annotations\Train\silver_quality\json_format\train_silver.json",r"data\Annotations\Dev\json_format\dev.json"]
fastText = None

def isGene(token:str):
    if token.isnumeric():
        return False
    if token.isalpha():
        return False
    for c in token:
        if c.isdigit():
            continue
        elif c.isalpha() and not c.isupper():
            return False
    return True

    
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    # if fastText == None:
    #     raise Exception("No fastText")
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.hasCapital()':word[0].capitalize() == word[0],
        'word.isdigit()': word.isdigit(),
        #'word.hasBioma()': "bio" in word,
        'word.isGene()': isGene(word),
        'postag': postag,
        'postag[:2]': postag[:2],
        'word.length()': len(word),
        'word.pos()':i,
    }
    # Add fasttext representation of word
    # for j in range(0,100):
    #     features.update({
    #         f'{j}:vec_fastText:':fastText.wv.get_vector(word)[j]
    #     })
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    # if i > 0 and i < len(sent)-1:
    #     features.update({'-1,+1:fastText()': fastText.wv.most_similar(positive=[sent[i-1][0],sent[i+1][0]],topn=1)[0][0]})
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

class Document:
    def __init__(self,id,title,abstract,year,journal,authors):
        self._id = id
        self.title = title
        self.abstract = abstract
        self._year= year
        self._journal = journal
        self._authors = authors
        self._entities = []
        self._ner = []

    def __str__(self):

        returned=f"{self._id},{self.title},entities:"
        for entity in self._entities:
            returned += str(entity)
            returned += "\n"
        return returned
    
    def add_entity(self,entity):
        if type(entity) != Entity:
            raise Exception("Not an entity")
        self._entities.append(entity)
    
    def ground_trouth(self):
        if len(self._entities) == 0:
            raise Exception("Cannot call ground trouth in a document without parsed entities")    
        print("Ground thruth of doc ",self._id)    
        i=0
        ner_tags = []
        title_entities = [ent for ent in self._entities if ent._location == "title"]
        abstract_entities = [ent for ent in self._entities if ent._location == "abstract"]
        i = 0
        text = self.title
        for entity in title_entities:
            before=nltk.word_tokenize(text[i:entity.start_idx])
            before=nltk.pos_tag(before)
            for token in before:
                tupla = (token[0],token[1],'O')
                ner_tags.append(tupla)
            span=nltk.word_tokenize(text[entity.start_idx:entity.end_idx+1])
            span=nltk.pos_tag(span)
            for idx,token in enumerate(span):
                if idx == 0:
                    tupla = (token[0],token[1],f"B-{entity.label}")
                else:
                    tupla = (token[0],token[1],f"I-{entity.label}")
                ner_tags.append(tupla)
            i = entity.end_idx + 1 
        #From last title entity to end of title
        after=nltk.word_tokenize(text[i:])
        after=nltk.pos_tag(after)
        for token in after:
            tupla = (token[0],token[1],'O')
            ner_tags.append(tupla)
        text = self.abstract
        i = 0
        for entity in abstract_entities:
            before=nltk.word_tokenize(text[i:entity.start_idx])
            before=nltk.pos_tag(before)
            for token in before:
                tupla = (token[0],token[1],'O')
                ner_tags.append(tupla)
            span=nltk.word_tokenize(text[entity.start_idx:entity.end_idx+1])
            span=nltk.pos_tag(span)
            for idx,token in enumerate(span):
                if idx == 0:
                    tupla = (token[0],token[1],f"B-{entity.label}")
                else:
                    tupla = (token[0],token[1],f"I-{entity.label}")
                ner_tags.append(tupla)
            i = entity.end_idx + 1 
        #From last title entity to end of title
        after=nltk.word_tokenize(text[i:])
        after=nltk.pos_tag(after)
        for token in after:
            tupla = (token[0],token[1],'O')
            ner_tags.append(tupla)
        self._ner=ner_tags
    
    @property
    def ner(self):
        if len(self._ner)==0:
            print("invoking indirect ground trouth")
            self.ground_trouth()
        return self._ner
                
class PredictedDocument(Document):
    from nltk.tokenize import TreebankWordTokenizer
    def __init__(self,document:Document=None):
        if document == None:
            super().__init__()
        else:
            super().__init__(document._id,document.title,document.abstract,document._year,document._journal,document._authors)
        self.pred_entities = []


    def predict_entities(self,model):
        raw_pred = []
        tokenizer = nltk.tokenize.TreebankWordTokenizer()

        for loc in locations:
            if loc == "abstract":
                text = self.abstract
            else:
                text = self.title
            #Fix with treebank
            tokenized=tokenizer.tokenize(text)
            spans = tokenizer.span_tokenize(text)
            to_analyze=nltk.pos_tag(tokenized)
            x_features=sent2features(to_analyze)
            curr_pred=model.predict([x_features])
            start_span = -1
            curr_pred = curr_pred[0]
            last_pred = None
            for token,span,pred in zip(tokenized,spans,curr_pred):
                # Started to recognise a new entity
                if  "B-" in pred and start_span == -1:
                    # Set start span to extract text
                    start_span = span[0]
                    end_span = span[1]
                    last_pred = pred
                    continue
                if "I-" in pred and start_span != -1:
                    end_span = span[1]
                if start_span != -1 and "I-" not in pred:
                    raw_pred.append(last_pred)
                    self.pred_entities.append(Entity(start_span,end_span,loc,text[start_span:end_span],last_pred[2:]))
                    start_span = -1
                    last_pred = None
    def out_entities(self):
        if len(self.pred_entities) == 0:
            raise Exception(f"No entities found in doc {self._id}")
        entities = [ent.to_dict() for ent in self.pred_entities]
        return entities
    
class Entity:
    locations = ["title","abstract"]

    def __init__(self,start_idx,end_idx,location,text_span,label):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.location = location
        self.text_span = text_span
        self.label= label
    
    @property
    def start_idx(self):
        return self._start_idx
    
    @property
    def location(self):
        return self._location
    
    @property
    def label(self):
        return self._label

    @start_idx.setter
    def start_idx(self, start_idx):
        if start_idx < 0:
            raise Exception("Index negative")
        self._start_idx = start_idx

    @location.setter
    def location(self,location):
        if location not in self.locations:
            raise Exception(f"{location} No location")
        self._location = location
    
    @label.setter
    def label(self,label):
        if label not in entity_labels:
            raise Exception(f"{label} Not in entity_labels")
        self._label = label
    def __str__(self):
        return f"TextSpan:{self.text_span} --- Label:{self.label} --- Idxs :{self.start_idx}-{self.end_idx} -- Location:{self.location}"
    def to_dict(self):
        return {'start_idx': self.start_idx, 'end_idx': self.end_idx-1, "location":self.location, "text_span":self.text_span, "label":self.label}
class Parser:
    """Class used to parse .json files and get respective document and entities"""
    _obj = None
    docs = []
    """This variable shall contain all the parsed Documents using decode_doc()"""
    def __init__(self):
        pass
    
    def decode_doc(self,filepath):
        """
        This function shall parse all the documents and put them inside Class variable
        """
        with open(filepath) as f:
            self._obj=json.load(f)
        # Analyzing parsed object
        print("I will now add ",len(self._obj.keys())," documents")
        for key in self._obj.keys():
            # Parsing every doc in json file
            doc = self._obj[key]
            # Every bit of information about every single document
            metadata = doc['metadata']
            entities = doc['entities']
            relations = doc['relations']
            binary_tag = doc['binary_tag_based_relations']
            ternary_tag = doc['ternary_tag_based_relations']
            ternary_mention = doc['ternary_mention_based_relations']
            # Create document 
            d = Document(key,metadata['title'],metadata['abstract'],metadata['year'],metadata['journal'],metadata['author'])
            # Add entities to document
            for value in entities:
                d.add_entity(Entity(value['start_idx'],value['end_idx'],value['location'],value['text_span'],value['label']))
            
            # At the end of the document parsing add it to the docs list
            self.docs.append(d)
        
    def prepare_crf(self,documents):
        """
        Function that gets documents paths and returns X,Y where X is a list of dictionary of features, Y is the label of the token
        ## Return:
        - X List of Dict, features, regarding current documents 
        - Y List of strings, labels of documents
        """
        for doc in documents:
            self.decode_doc(doc)
        return self.engeneere_features()

    def engeneere_features(self):
        X = []
        Y = []
        for doc in self.docs:
            x_temp = []
            y_temp = []
            for i in range(0,len(doc.ner)):
                x_temp.append(word2features(doc.ner,i))
            for elem in doc.ner:
                y_temp.append(elem[2])
            X.append(x_temp)
            Y.append(y_temp)
        return X,Y

    def prepare_train(self):
        '''Resets class variable docs and replace it with train collection'''
        self.docs = []
        return self.prepare_crf(documents_train)
    
    def prepare_dev(self):
        '''Resets class variable docs and replace it with dev collection'''
        self.docs = []
        return self.prepare_crf(documents_dev)
    
    def __str__(self):
        return str(self._obj)


def performance_on_datasets(p:Parser,model,docs:list[str]):
    p.docs = []
    averages=["micro","macro"]
    with open("perf.txt","w") as f:
        for doc in docs:
            X,Y = p.prepare_crf([doc])
            y_predict=model.predict(X)
            f.write(f"Performances on {doc}:\n")
            for avg in averages:
                f.write(f"Precision {avg}:{sklearn_crfsuite.metrics.flat_precision_score(Y,y_predict,average=avg)}\n")
                f.write(f"Recall {avg}:{sklearn_crfsuite.metrics.flat_recall_score(Y,y_predict,average=avg)}\n")
                f.write(f"F1-score {avg}:{sklearn_crfsuite.metrics.flat_f1_score(Y,y_predict,average=avg)}\n")
            p.docs = []
        
