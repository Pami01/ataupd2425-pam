from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import load_dataset
from Logic import Entity, Document, PredictedDocument, Parser
import torch
import json
from transformers import pipeline
import evaluate
import os
# 2. Label Mapping
labels_list = ["O"]
locations = ["title","abstract"]
#labels_list = ["O", "B-Drug", "I-Drug", "B-Disease", "I-Disease", "B-Gene"]

for elem in ["anatomical location","animal","biomedical technique","bacteria","chemical","dietary supplement","DDF","drug","food","gene","human","microbiome","statistical technique"]:
    labels_list.append(f"B-{elem}")
    labels_list.append(f"I-{elem}")

label_to_id = {label: i for i, label in enumerate(labels_list)}
id_to_label = {i: label for label, i in label_to_id.items()}



DOC_PATH = [r"C:\Users\xLoll\Desktop\GutBrainIE2025\data\Annotations\Dev\json_format\dev.json"]
# Read documents
p = Parser()
for path in DOC_PATH:
    p.decode_doc(path)

dirs=os.listdir(r"C:\Users\xLoll\Desktop\GutBrainIE2025\models")

model_name = "./BiomedNLP-PubMedBERT-base-uncased-abstract-12-CW-xtreme"
# Carica
for model_name in dirs:
    try:
        print(f"Evaluating {model_name}")
        model = AutoModelForTokenClassification.from_pretrained(f"./models/{model_name}")
        tokenizer = AutoTokenizer.from_pretrained(f"./models/{model_name}")
        tokenizer.model_max_length=512

        # Crei la pipeline
        nlp = pipeline(
            "token-classification", 
            model=model, 
            tokenizer=tokenizer, 
            aggregation_strategy="first",
            device=0  # importante!
        )

        # Use PredictedDocuments to add entities
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
                    pd.pred_entities.append(Entity(start,end,loc,test_sentence[start:end],label))
                    # print(f"{word} --> {label} -> {start},{end}")
            pds.append(pd)
            
        # Print out entities usin PDs functions
        to_print = {}
        for pd in pds:
            curr_doc = {f"{pd._id}":{"entities":pd.out_entities()}}
            to_print.update(curr_doc)
        with open(f"./runs/{model_name}.json", "w") as f:
            json.dump(to_print,f)


        prec,recall,f1,mp,mr,mf=evaluate.eval_submission_6_1_NER(f"./runs/{model_name}.json")
        with open("evaluate_performances.txt","a") as f:
            f.write(f"{model_name}\n")
            f.write(f"p:{prec},r:{recall},f1:{f1},micro p:{mp},micro r:{mr},micro f1:{mf}\n")
    except:
        print(f"Something went wrong with file {model_name}")