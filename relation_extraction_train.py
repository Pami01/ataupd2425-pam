import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
import random 
import copy
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
LABELS = ["no relation","influence", "is linked to", "target", "located in","impact","change abundance","change effect","used by","affect","part of","interact","produced by","strike","change expression","administered","is a","compared to"]  # <-- modifica in base al tuo dataset
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

# Load data
dataset = []
norel = []
TRAIN_PATH = [r"C:\Users\xLoll\Desktop\GutBrainIE2025\data\Annotations\Dev\json_format\dev.json",r"C:\Users\xLoll\Desktop\GutBrainIE2025\data\Annotations\Train\bronze_quality\json_format\train_bronze.json",r"data\Annotations\Train\gold_quality\json_format\train_gold.json",r"data\Annotations\Train\platinum_quality\json_format\train_platinum.json",r"data\Annotations\Train\silver_quality\json_format\train_silver.json"]

# FILES_PATH = [r"C:\Users\xLoll\Desktop\GutBrainIE2025\data\Annotations\Dev\json_format\dev.json",]
for file in TRAIN_PATH:
    with open(file,"r") as f:
        raw_data = json.load(f)
    # Add marker to text for training 
    # Raw data is key - value where key is document number
    for item in raw_data.keys():
        # Effective document info
        item = raw_data[item]
        # Metadata of document
        metadata = item["metadata"]
        relations = []
        #Add real relations, labeled and precise
        for rel in item["relations"]:
            loc = rel["subject_location"]
            text = metadata[loc]
            s_start, s_end = rel["subject_start_idx"], rel["subject_end_idx"]+1
            o_start, o_end = rel["object_start_idx"], rel["object_end_idx"]+1
            label = rel["predicate"] if rel["predicate"] in LABELS else "ERROR"
            
            relations.append({"loc":loc,"s_start":s_start, "s_end":s_end,"o_start":o_start, "o_end":o_end})

            if s_start < o_start:
                marked_text = (
                    text[:s_start] + "[E1]" + text[s_start:s_end] + "[/E1]" +
                    text[s_end:o_start] + "[E2]" + text[o_start:o_end] + "[/E2]" +
                    text[o_end:]
                )
            else:
                marked_text = (
                    text[:o_start] + "[E2]" + text[o_start:o_end] + "[/E2]" +
                    text[o_end:s_start] + "[E1]" + text[s_start:s_end] + "[/E1]" +
                    text[s_end:]
                )

            dataset.append({"text": marked_text, "label": label2id[label]})
        
        #Now add noisy ones, combining entities with no relations between them
        entities = item["entities"]
        for i, subj in enumerate(entities):
            for j, obj in enumerate(entities):
                if i == j:
                    continue

                loc_subj = subj["location"]
                loc_obj = obj["location"]
                if loc_subj != loc_obj:
                    continue
                s_start, s_end = subj["start_idx"], subj["end_idx"]+1
                o_start, o_end = obj["start_idx"], obj["end_idx"]+1

                check = {"loc":loc,"s_start":s_start, "s_end":s_end,"o_start":o_start, "o_end":o_end}

                if check in relations:
                    continue
                # Add marker into text
                if s_start < o_start:
                    marked_text = f"{text[:s_start]}[E1]{text[s_start:s_end]}[/E1]{text[s_end:o_start]}[E2]{text[o_start:o_end]}[/E2]{text[o_end:]}"
                else:
                    marked_text = f"{text[:o_start]}[E2]{text[o_start:o_end]}[/E2]{text[o_end:s_start]}[E1]{text[s_start:s_end]}[/E1]{text[s_end:]}"

                label = "no relation"
                #If relation is not exhisting, add this to a list
                norel.append({"text": marked_text, "label": label2id[label]})

# NEGATIVE SAMPLING (SMART)
#Get number of real relations and proportionally get a no relation
base_dataset=copy.deepcopy(dataset)

for propNumber in [1,2,3]:
    # propNumber = 2
    try:
        dataset = []
        dataset.extend(base_dataset)
        rLen = len(dataset)
        #Extend the dataset a sample from the norel proportional to propNumber and rLen
        dataset.extend(random.sample(norel,propNumber*rLen))
                    
        train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=42)
        dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
    except Exception as e:
        print(f"Dataset exception catched, aborted training {str(e)}")
        continue
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.add_special_tokens({"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]})
    except:
        print("Problem loading tokenizer")
        continue

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )
    try:
        dataset = dataset.map(tokenize)
        val_dataset = val_dataset.map(tokenize)
    except Exception as e:
        print(f"Problem with preprocessing dataset {str(e)}")
        continue

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=id2label, label2id=label2id
    )
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir="./re_output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        resume_from_checkpoint=True
    )

    # Base metrics of steps in runs
    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        labels = p.label_ids
        acc = (preds == labels).mean()
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    for i in range(2):
        try:
            trainer.train()
        except:
            print("Problem with training")
            continue
        model_name = f"RE-BiomedNLP-{propNumber}NoRel-{i}epoch-COMPLETE_DATASET"
        model.save_pretrained(model_name)
        tokenizer.save_pretrained(model_name)

