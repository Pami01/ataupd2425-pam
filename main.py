import json
import nltk 
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn_crfsuite
import pickle
from sklearn_crfsuite import metrics
from  collections import Counter

entity_labels = ["anatomical location","animal","biomedical technique","bacteria","chemical","dietary supplement","DDF","drug","food","gene","human","microbiome","statistical technique"]
locations = ["title","abstract"]

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
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
        i=0
        ner_tags = []
        for loc in locations:
            if loc == "abstract":
                text = self.abstract
            else:
                text = self.title
            for entity in self._entities:
                if entity.location == loc:
                    before=nltk.word_tokenize(text[i:entity.start_idx])
                    before=nltk.pos_tag(before)
                    for token in before:
                        tupla = (token[0],token[1],'O')
                        ner_tags.append(tupla)
                    span=nltk.word_tokenize(text[entity.start_idx:entity.end_idx+1])
                    span=nltk.pos_tag(span)
                    for token in span:
                        tupla = (token[0],token[1],entity.label)
                        ner_tags.append(tupla)
                    i = entity.end_idx + 1 
                else:
                    continue
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

    def __str__(self):
        return str(self._obj)

p = Parser()

load = True

if load == True:
    obj=pickle.load(open("model.pickle",'rb'))
    text = "Hypothesis of a potential BrainBiota and its relation to CNS autoimmune inflammation, eat by rodents and used by antidepressants."
    tokens = nltk.word_tokenize(text)
    words = nltk.pos_tag(tokens)

    def print_state_features(state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-8s %s" % (weight, label, attr))

    print("Top positive:")
    print_state_features(Counter(obj.state_features_).most_common(30))

    print("\nTop negative:")
    print_state_features(Counter(obj.state_features_).most_common()[-30:])
    feat = obj.state_features_
    sorted=dict(sorted(feat.items(),key=lambda item: item[1]))
    #print(sorted)
    print(sent2features(words))
    print(obj.predict([sent2features(words)]))
else:
    documents = [r"data\Annotations\Train\bronze_quality\json_format\train_bronze.json",r"data\Annotations\Train\gold_quality\json_format\train_gold.json",r"data\Annotations\Train\platinum_quality\json_format\train_platinum.json",r"data\Annotations\Train\silver_quality\json_format\train_silver.json"]
    X = []
    Y = []
    for docpath in documents:
        p.decode_doc(docpath)
        for doc in p.docs:
            x_temp = []
            y_temp = []
            for i in range(0,len(doc.ner)):
                x_temp.append(word2features(doc.ner,i))
            for elem in doc.ner:
                y_temp.append(elem[2])
            X.append(x_temp)
            Y.append(y_temp)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    crf = sklearn_crfsuite.CRF(algorithm='lbfgs',
                            c1=0.1,
                            c2=0.1,
                            max_iterations=100,
                            # min_freq=30,
                            all_possible_transitions=True,
                            verbose=True)

    crf.fit(X_train, y_train)

    pickle.dump(crf,open("model.pickle",'wb+'))
    text = "Hypothesis of a potential BrainBiota and its relation to CNS autoimmune inflammation."
    tokens = nltk.word_tokenize(text)
    words = nltk.pos_tag(tokens)

    # print(sent2features(words))
    #print(crf.predict([sent2features(words)]))

    print(crf.predict_marginals([sent2features(words)]))
    print(crf.predict([sent2features(words)]))
    feat=crf.state_features_
    sorted=dict(sorted(feat.items(),key=lambda item: item[1]))
    print(sorted)