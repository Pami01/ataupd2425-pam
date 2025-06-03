import pickle
from collections import Counter
import sklearn_crfsuite
import sklearn_crfsuite.metrics
from Logic import Parser, PredictedDocument, sent2features, word2features
from sklearn.model_selection import train_test_split
import json
import nltk
from collections import Counter

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))


def print_state_features(state_features,label_filter=None,attr_filter=None):
            states = []
            for (attr, label), weight in state_features:                    
                if label_filter is not None and label_filter != label:
                    continue
                if attr_filter is not None and attr_filter not in attr:
                    continue
                states.append("%0.6f %-8s %s" % (weight, label, attr))
            return  states


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

#If load = True, the program will load the pickle file and predict labels
load = True
#filname that will be used for outputting prediction of NER
fileDump = "out.json"

p = Parser()
#All documents path
documents = [r"data\Annotations\Train\bronze_quality\json_format\train_bronze.json",r"data\Annotations\Train\gold_quality\json_format\train_gold.json",r"data\Annotations\Train\platinum_quality\json_format\train_platinum.json",r"data\Annotations\Train\silver_quality\json_format\train_silver.json",r"data\Annotations\Dev\json_format\dev.json"]
dev = r"data\Annotations\Dev\json_format\dev.json"
platinum = [r"data\Annotations\Train\platinum_quality\json_format\train_platinum.json"]
gold = [r"data\Annotations\Train\gold_quality\json_format\train_gold.json"]
test_set = r"C:\Users\xLoll\Desktop\GutBrainIE_2025_Test_Data\articles_test.json"
if load == True:
    obj=pickle.load(open("model-withB-I.pickle",'rb'))

    #Watch out using decode_val_doc or decode_doc, see docs (different format of docs needs different functions)
    p.decode_doc(dev)
    
    to_print = {}
    for doc in p.docs:
        pd=PredictedDocument(document=doc)
        pd.predict_entities(obj)
        curr_doc = {f"{doc._id}":{"entities":pd.out_entities()}}
        to_print.update(curr_doc)

    with open(fileDump, "w") as f:
        json.dump(to_print,f)
    print("Top likely transitions:")
    print_transitions(Counter(obj.transition_features_).most_common(20))

    print("\nTop unlikely transitions:")
    print_transitions(Counter(obj.transition_features_).most_common()[-20:])

    feat=obj.state_features_
    sorted=dict(sorted(feat.items(),key=lambda item: item[1],reverse=True)[:20])
    print("\nTop 20 features:")
    for key,value in sorted.items():
         print(key,value)
else:
    X,Y = p.prepare_train()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.01, random_state=42)

    crf = sklearn_crfsuite.CRF(algorithm='lbfgs',
                            c1=1e-6,
                            c2=1e-6,
                            #max_iterations=30,
                            # min_freq=30,
                            all_possible_transitions=True,
                            verbose=True)
    #Fit CRF model
    crf.fit(X_train, y_train)
    
    #Save crf model as pickle file
    pickle.dump(crf,open("model-withB-I-OVERFIT-NoDevBronze.pickle",'wb+'))
    
    feat=crf.state_features_
    sorted=dict(sorted(feat.items(),key=lambda item: item[1]))
    print(sorted)

