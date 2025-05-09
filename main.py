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




p = Parser()
#p.decode_doc("test.json")
#doc=p.docs[0]
# true labels 
#ner = doc.ner
documents = [r"data\Annotations\Train\bronze_quality\json_format\train_bronze.json",r"data\Annotations\Train\gold_quality\json_format\train_gold.json",r"data\Annotations\Train\platinum_quality\json_format\train_platinum.json",r"data\Annotations\Train\silver_quality\json_format\train_silver.json",r"data\Annotations\Dev\json_format\dev.json"]
#documents = [r"data\Annotations\Dev\json_format\dev.json"]
#y_labels = [elem[2] for elem in ner]
load = False


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
    obj=pickle.load(open("model-withB-I-OVERFIT.pickle",'rb'))
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
    print(sklearn_crfsuite.metrics.flat_classification_report(Y,y_predict))

    for avg in averages:
        print(f"Precision {avg}:{sklearn_crfsuite.metrics.flat_precision_score(Y,y_predict,average=avg)}")
        print(f"Recall {avg}:{sklearn_crfsuite.metrics.flat_recall_score(Y,y_predict,average=avg)}")
        print(f"F1-score {avg}:{sklearn_crfsuite.metrics.flat_f1_score(Y,y_predict,average=avg)}")
    
    
    feat = Counter(obj.state_features_).most_common()
    states = print_state_features(feat,attr_filter="fastText()")[:50]
    for state in states:
        print(state)
    sent_feat=sent2features(words)
    

    prediction = obj.predict([sent2features(words)])
    for token,label in zip(tokens,prediction[0]):
        if label != 'O':
            print(token,label)


    #Evaluate performance of model in every corpora
    #performance_on_datasets(p,obj,documents)
    to_print = {}
    for doc in p.docs:
        pd=PredictedDocument(document=doc)
        pd.predict_entities(obj)
        curr_doc = {f"{doc._id}":{"entities":pd.out_entities()}}
        to_print.update(curr_doc)
    with open("out.json", "w") as f:
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

    crf.fit(X_train, y_train)
    
    pickle.dump(crf,open("model-withB-I-OVERFIT-NoDevBronze.pickle",'wb+'))
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

