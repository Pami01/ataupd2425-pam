from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

# 1. MODELLO & TOKENIZER
model_checkpoint = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# 2. DATASET NER (usa CONLL2003 per esempio)
raw_datasets = load_dataset("conll2003",trust_remote_code=True)

# 3. LABELS
label_list = raw_datasets["train"].features["ner_tags"].feature.names
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}

# 4. MODELLLO con numero corretto di label
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)

# 5. FUNZIONE di tokenizzazione + allineamento etichette
def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True
    )
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    labels = []

    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(example["ner_tags"][word_idx])
        else:
            labels.append(-100)
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# 6. PREPROCESSA i dati
tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=False)

# 7. METRICHE
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_preds = [
        [id2label[p] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
        "accuracy": accuracy_score(true_labels, true_preds),
    }

# 8. DATA COLLATOR (gestisce padding dinamico e batch uniformi)
data_collator = DataCollatorForTokenClassification(tokenizer)

# 9. ARGOMENTI di training
args = TrainingArguments(
    output_dir="./biobert-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 10. TRAINER
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 11. ADDRESTRAMENTO
trainer.train()

# 12. SALVA il modello
trainer.save_model("./biobert-ner-finetuned")
tokenizer.save_pretrained("./biobert-ner-finetuned")

