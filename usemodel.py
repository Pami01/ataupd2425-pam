from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import load_dataset
from Logic import Entity, Document, PredictedDocument, Parser
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

def writeMETA(filepath,taskID,runID,training,preprocessing,tdu,dor):
    with open(filepath,"w") as f:
        out = f"""Team ID: ataupd2425-pam
Task ID: {taskID}
Run ID: {runID}
Training: {training}
PreProcessing: {preprocessing}
Training data used: {tdu}
Details of the run: {dor}
https://github.com/Pami01/ataupd2425-pam
"""    
        f.write(out)
def loadModel(model_name):
    global model
    global tokenizer
    global nlp
    print(f"Evaluating {model_name}")
    model = AutoModelForTokenClassification.from_pretrained(f"{base_dir}/{model_name}")
    tokenizer = AutoTokenizer.from_pretrained(f"{base_dir}/{model_name}")
    tokenizer.model_max_length=512

    # Crei la pipeline
    nlp = pipeline(
        "token-classification", 
        model=model, 
        tokenizer=tokenizer, 
        aggregation_strategy="first",
        device=0  # importante!
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

DOC_PATH = [r"C:\Users\xLoll\Desktop\GutBrainIE2025\data\Annotations\Dev\json_format\dev.json"]
# Read documents
p = Parser()
for path in DOC_PATH:
    p.decode_doc(path)

base_dir = r"C:\Users\xLoll\Desktop\GutBrainIE2025\models_evaluate_dev"
dirs=os.listdir(base_dir)
# dirs = ["NuNerv2.0-10-CW-xtreme"]
model_name = "./BiomedNLP-PubMedBERT-base-uncased-abstract-12-CW-xtreme"
# Carica
runID = 0
#NER TASK with BERT models
for model_name in dirs:
    try:
        runID+=1
        loadModel(model_name)
        # Use PredictedDocuments to add entities
        pds = predictDocs(p)
        # Print out for evaluation script!
        ner = dumpNER(pds)
        try:
            os.mkdir(f"./evaluation/ataupd2425-pam_T61_{runID}_{model_name}")
        except Exception as e:
            print(f"Something went wrong creating directory ./evaluation/ataupd2425-pam_T61_{runID}_{model_name}")    
        with open(f"./evaluation/ataupd2425-pam_T61_{runID}_{model_name}/ataupd2425-pam_T61_{runID}_{model_name}.json", "w") as f:
                # Dump file for evaluation script
                json.dump(ner,f)
        prec,recall,f1,mp,mr,mf=evaluate.eval_submission_6_1_NER(f"./evaluation/ataupd2425-pam_T61_{runID}_{model_name}/ataupd2425-pam_T61_{runID}_{model_name}.json")
        with open("evaluate_performances.txt","a") as f:
             f.write(f"{model_name}\n")
             f.write(f"p:{prec},r:{recall},f1:{f1},micro p:{mp},micro r:{mr},micro f1:{mf}\n")
        writeMETA(f"./evaluation/ataupd2425-pam_T61_{runID}_{model_name}/ataupd2425-pam_T61_{runID}_{model_name}.meta","T61",runID,f"Fine tuning using HuggingFace pipeline of {model_name}, usage of Custom Weights, inversionally proportional to number of weights (more loss to less frequent labels) ","Parsing of data in the datasets, conversion of every document entity into a list of tokens and labels","Platinum,Gold,Silver","TBD")        
        #CREATE ENTITIES WITH NER
        dump = {}
        for doc in pds:
            metadata = {"title":doc.title,"author":doc._authors,"journal":doc._journal,"year":doc._year,"abstract":doc.abstract,}
            entities = doc.out_entities()
            doc_info = {"metadata":metadata,"entities":entities}
            dump.update({f"{doc._id}":doc_info})
        with open(f"./evaluation/ataupd2425-pam_T61_{runID}_{model_name}/tempNER.json","w") as f:
            json.dump(dump,f)            
    except Exception as e:
        print(f"Something went wrong with file {model_name}, {str(e)}")

#RE Task 

# RE = ["./RE-BiomedNLP-2NoRel-5epoch",]
# for model in RE:
#     runID += 1
#     use_relation_extraction.loadModel(model)
#     try:
#         use_relation_extraction.print_RE(r"C:\Users\xLoll\Desktop\GutBrainIE2025\evaluation\ataupd2425-pam_T61_1_scibert-10\tempNER.json","./evaluation/RE",model[2:],runID)
#     except Exception as e:
#         print(f"Error printing RE {model}, {str(e)}")