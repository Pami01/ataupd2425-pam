from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import load_dataset
from Logic import Entity, Document, PredictedDocument, Parser, writeMETA
import torch
import json
from transformers import pipeline
import evaluate
import os
import use_relation_extraction
# 2. Label Mapping
labels_list = ["O"]
locations = ["title","abstract"]
#labels_list = ["O", "B-Drug", "I-Drug", "B-Disease", "I-Disease", "B-Gene"]
char_to_remove = [",",".",";",":","!","?"," ","(",")"]

for elem in ["anatomical location","animal","biomedical technique","bacteria","chemical","dietary supplement","DDF","drug","food","gene","human","microbiome","statistical technique"]:
    labels_list.append(f"B-{elem}")
    labels_list.append(f"I-{elem}")

label_to_id = {label: i for i, label in enumerate(labels_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

model = None
tokenizer = None
nlp = None


def loadModel(model_name):
    global model
    global tokenizer
    global nlp
    print(f"Evaluating {model_name}")
    model = AutoModelForTokenClassification.from_pretrained(f"{model_name}")
    tokenizer = AutoTokenizer.from_pretrained(f"{model_name}")
    tokenizer.model_max_length=512

    # Crei la pipeline
    nlp = pipeline(
        "token-classification", 
        model=model, 
        tokenizer=tokenizer, 
        aggregation_strategy="first",
        device=0
    )

def predictDocs(p:Parser):
    pds = []
    for doc in p.docs:
        pd = PredictedDocument(doc)
        for loc in locations:
            test_sentence=getattr(doc,loc)
            results = nlp(test_sentence)
            for res in results:
                word = res["word"]
                label = res["entity_group"]
                start = res["start"]
                end = res["end"]
                text_to_append = test_sentence[start:end]
                if text_to_append in char_to_remove:
                    continue
                # TODO Fix this, trailing and starting chars/spaces
                # Starting forbidden chars
                starting_word = text_to_append
                try:
                    while text_to_append[0] in char_to_remove:
                        start+=1
                        text_to_append = text_to_append[1:]
                    while text_to_append[-1] in char_to_remove:
                        end-=1
                        text_to_append=text_to_append[:-1]
                    if start >= end or len(text_to_append)<=1:
                        continue
                    pd.pred_entities.append(Entity(start,end,loc,test_sentence[start:end],label))
                except:
                    #If something goes wrong with the entity prediction (probably only one character) print it out and skip the entity
                    print(f"Probably something went wrong here {doc._id}, at indexes {start}:{end}")
                # print(f"{word} --> {label} -> {start},{end}")
        pds.append(pd)
    return pds

def dumpNER(pds):
    to_print = {}
    for pd in pds:
        curr_doc = {f"{pd._id}":{"entities":pd.out_entities()}}
        to_print.update(curr_doc)
    return to_print

DOC_PATH = [r"C:\Users\xLoll\Desktop\GutBrainIE_2025_Test_Data\articles_test.json"]
# Read documents
p = Parser()
for path in DOC_PATH:
    p.decode_val_doc(path)

base_dir = ""
model_name = "./BiomedNLP-PubMedBERT-base-uncased-abstract-12-CW-xtreme"
runID = 0
#NER TASK with BERT models
dirs = [r"./models/biobert-base-cased-v1.2-14-CW-xtreme",r"./models/biosyn-sapbert-bc2gn-8",r"./models/biosyn-sapbert-bc2gn-12",r"./models/saved_model",r"./models/NuNerv2.0-22-CW-xtreme",r"./models/scibert-47",r"./models/scibert-27"]
for model_path in dirs:
    try:
        runID+=1
        loadModel(model_path)
        model_name = model_path[9:]
        # Use PredictedDocuments to add entities
        pds = predictDocs(p)
        # Print out for evaluation script!
        ner = dumpNER(pds)

        current_dir_name = f"./delievery/ataupd2425-pam_T61_{runID}_{model_name}"
        try:
            os.mkdir(current_dir_name)
        except Exception as e:
            print(f"Something went wrong creating directory {current_dir_name}")    
        with open(f"{current_dir_name}/ataupd2425-pam_T61_{runID}_{model_name}.json", "w") as f:
                # Dump file for evaluation script
                json.dump(ner,f)
        # prec,recall,f1,mp,mr,mf=evaluate.eval_submission_6_1_NER(f"{current_dir_name}/ataupd2425-pam_T61_{runID}_{model_name}.json")
        # with open("evaluate_performances.txt","a") as f:
        #      f.write(f"{model_name}\n")
        #      f.write(f"p:{prec},r:{recall},f1:{f1},micro p:{mp},micro r:{mr},micro f1:{mf}\n")
        writeMETA(f"{current_dir_name}/ataupd2425-pam_T61_{runID}_{model_name}.meta","T61",runID,f"Fine tuning using HuggingFace pipeline of {model_name}, usage of Custom Weights, inversionally proportional to number of weights (more loss to less frequent labels) ","Parsing of data in the datasets, conversion of every document entity into a list of tokens and labels","Platinum,Gold,Silver,Bronze,Dev","TBD")        
        #CREATE ENTITIES WITH NER
        dump = {}
        for doc in pds:
            metadata = {"title":doc.title,"author":doc._authors,"journal":doc._journal,"year":doc._year,"abstract":doc.abstract,}
            entities = doc.out_entities()
            doc_info = {"metadata":metadata,"entities":entities}
            dump.update({f"{doc._id}":doc_info})
        with open(f"{current_dir_name}/tempNER.json","w") as f:
            json.dump(dump,f)            
    except Exception as e:
        print(f"Something went wrong with file {model_name}, {str(e)}")

#RE Task 
RE = ["./RE-BiomedNLP-1NoRel-1epoch-COMPLETE_DATASET","./RE-BiomedNLP-2NoRel-1epoch-COMPLETE_DATASET","./RE-BiomedNLP-3NoRel-1epoch-COMPLETE_DATASET"]
#PATHS OF EXTRACTED NER 
NER = []
# for each model in RE
runID = 0
for model in RE:
    # for each of the selected NER runs (different models)
    for ner_run in NER:
        use_relation_extraction.loadModel(model)
        try:
            use_relation_extraction.print_RE(ner_run,"./delievery",model[2:],runID)
        except Exception as e:
            print(f"Error printing RE {model}, {str(e)}")
        runID += 1