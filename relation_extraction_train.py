import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
LABELS = ["influence", "is linked to", "target", "located in","impact","change abundance","change effect","used by","affect","part of","interact","produced by","strike","change expression","administered","is a","compared to"]  # <-- modifica in base al tuo dataset
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

# Load data
with open(r"C:\Users\xLoll\Desktop\GutBrainIE2025\data\Annotations\Dev\json_format\dev.json") as f:
    raw_data = json.load(f)

examples = []

# Add marker to text for training 
# Raw data is key - value where key is document number
for item in raw_data.keys():
    # Effective document info
    item = raw_data[item]
    # Metadata of document
    metadata = item["metadata"]
    for rel in item["relations"]:
        loc = rel["subject_location"]
        text = metadata[loc]
        s_start, s_end = rel["subject_start_idx"], rel["subject_end_idx"]+1
        o_start, o_end = rel["object_start_idx"], rel["object_end_idx"]+1
        label = rel["predicate"] if rel["predicate"] in LABELS else "no_relation"
        
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

        examples.append({"text": marked_text, "label": label2id[label]})

train_data, val_data = train_test_split(examples, test_size=0.1, random_state=42)
dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_special_tokens({"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]})

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

dataset = dataset.map(tokenize)
val_dataset = val_dataset.map(tokenize)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(LABELS), id2label=id2label, label2id=label2id
)
model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(
    output_dir="./re_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
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

trainer.train()