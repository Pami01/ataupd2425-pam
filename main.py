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
documents_train = [r"data\Annotations\Train\bronze_quality\json_format\train_bronze.json",r"data\Annotations\Train\gold_quality\json_format\train_gold.json",r"data\Annotations\Train\platinum_quality\json_format\train_platinum.json",r"data\Annotations\Train\silver_quality\json_format\train_silver.json"]
fastText = None

def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels

def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    #selected_indices = random.sample(indices, 25)
    selected_indices = []

    ## Select elements that are labeled as genes in ground trouth to verify if fastText is reliable
    for elem in genes_tokens:
        try:
            selected_indices.append(np.where(labels == elem)[0][0])
        except:
            print("Elem ",elem," not found in vocab")

    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]),color="red")
    plt.show()

def print_state_features(state_features,label_filter=None,attr_filter=None):
            states = []
            for (attr, label), weight in state_features:                    
                if label_filter is not None and label_filter != label:
                    continue
                if attr_filter is not None and attr_filter not in attr:
                    continue
                states.append("%0.6f %-8s %s" % (weight, label, attr))
            return  states

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
def explain(sent_feat,ents_labels):
        for index,feature in enumerate(sent_feat):
            print(f"Token n {index}")
            for label in ents_labels:
                print(label)
                sum = 0
                for entry,value in feature.items():
                    find = f"{entry}:{value}"
                    for elem in feat:
                        if find == elem[0][0] and elem[0][1] == label:
                            print(elem[0][0],elem[1])
                            sum+=elem[1]
                print(sum)


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
        for doc in p.docs:
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

p = Parser()
#p.decode_doc("test.json")
#doc=p.docs[0]
# true labels 
#ner = doc.ner
documents = [r"data\Annotations\Train\bronze_quality\json_format\train_bronze.json",r"data\Annotations\Train\gold_quality\json_format\train_gold.json",r"data\Annotations\Train\platinum_quality\json_format\train_platinum.json",r"data\Annotations\Train\silver_quality\json_format\train_silver.json"]
#documents = [r"data\Annotations\Dev\json_format\dev.json"]
#y_labels = [elem[2] for elem in ner]
load = True

import gensim
import nltk

# for doc in documents:
#     p.decode_doc(doc)
# total = []
# genes_tokens = []
# for doc in p.docs:
#     curr_doc = []
#     for entity in doc.ner:
#         curr_doc.append(entity[0])
#         if entity[2] == "gene":
#             genes_tokens.append(entity[0])
#     total.append(curr_doc)
# fastText = gensim.models.FastText(total,vector_size=100,sg=1)
# print(fastText.wv.most_similar(positive=["gene"],topn=3))

# x_vals, y_vals, labels = reduce_dimensions(word2vec)

# plot_with_matplotlib(x_vals, y_vals, labels)
platinum = [r"data\Annotations\Train\platinum_quality\json_format\train_platinum.json"]
gold = [r"data\Annotations\Train\gold_quality\json_format\train_gold.json"]
if load == True:
    obj=pickle.load(open("model-len-pos.pickle",'rb'))
    text = "(1) Background: studies have shown that some patients experience mental deterioration after bariatric surgery. (2) Methods: We examined whether the use of probiotics and improved eating habits can improve the mental health of people who suffered from mood disorders after bariatric surgery. We also analyzed patients' mental states, eating habits and microbiota. (3) Results: Depressive symptoms were observed in 45% of 200 bariatric patients. After 5 weeks, we noted an improvement in patients' mental functioning (reduction in BDI and HRSD), but it was not related to the probiotic used. The consumption of vegetables and whole grain cereals increased (DQI-I adequacy), the consumption of simple sugars and SFA decreased (moderation DQI-I), and the consumption of monounsaturated fatty acids increased it. In the feces of patients after RYGB, there was a significantly higher abundance of two members of the Muribaculaceae family, namely Veillonella and Roseburia, while those after SG had more Christensenellaceae R-7 group, Subdoligranulum, Oscillibacter, and UCG-005. (4) Conclusions: the noted differences in the composition of the gut microbiota (RYGB vs. SG) may be one of the determinants of the proper functioning of the gut-brain microbiota axis, although there is currently a need for further research into this topic using a larger group of patients and different probiotic doses."
    tokens = nltk.word_tokenize(text)
    words = nltk.pos_tag(tokens)

    X,Y=p.prepare_dev()
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    y_predict=obj.predict(X)
    
    for x,y,y_real in zip(X,y_predict,Y):
        for elem_x,predicted_label,real_label in zip(x,y,y_real):
            if predicted_label == "gene":
                print(elem_x['word.lower()']," ",predicted_label,real_label,"\n")

    averages=["micro","macro"]
    for avg in averages:
        print(f"Precision {avg}:{sklearn_crfsuite.metrics.flat_precision_score(Y,y_predict,average=avg)}")
        print(f"Recall {avg}:{sklearn_crfsuite.metrics.flat_recall_score(Y,y_predict,average=avg)}")
        print(f"F1-score {avg}:{sklearn_crfsuite.metrics.flat_f1_score(Y,y_predict,average=avg)}")
    
    
    feat = Counter(obj.state_features_).most_common()
    states = print_state_features(feat,attr_filter="fastText()")[:50]
    for state in states:
        print(state)
    sent_feat=sent2features(words)
    
    #explain(sent_feat,["O","microbiome"])

    prediction = obj.predict([sent2features(words)])
    for token,label in zip(tokens,prediction[0]):
        if label != 'O':
            print(token,label)
    # print(obj.predict_marginals([sent2features(words)]))
else:
    X,Y = p.prepare_train()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    crf = sklearn_crfsuite.CRF(algorithm='lbfgs',
                            c1=0.1,
                            c2=0.1,
                            #max_iterations=30,
                            # min_freq=30,
                            all_possible_transitions=True,
                            verbose=True)

    crf.fit(X_train, y_train)
    
    pickle.dump(crf,open("model-len-pos.pickle",'wb+'))
    text = "Hypothesis of a potential gut microbiota and its relation to CNS autoimmune inflammation."
    tokens = nltk.word_tokenize(text)
    words = nltk.pos_tag(tokens)

    # print(sent2features(words))
    #print(crf.predict([sent2features(words)]))

    print(crf.predict_marginals([sent2features(words)]))
    print(crf.predict([sent2features(words)]))
    feat=crf.state_features_
    sorted=dict(sorted(feat.items(),key=lambda item: item[1]))
    print(sorted)

