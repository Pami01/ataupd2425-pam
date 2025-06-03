from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import load_dataset
import torch
import torch.nn as nn
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import torch
import torch.nn as nn
from collections import Counter
import itertools
import torch

def preprocess(example):
    tokenized_inputs = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=512
    )
    
    word_ids = tokenized_inputs.word_ids()
    aligned_labels = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            aligned_labels.append(label_to_id[example["labels"][word_idx]])
        else:
            aligned_labels.append(label_to_id[example["labels"][word_idx]])
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(-1)

    true_labels = []
    true_predictions = []

    for pred, label in zip(predictions, labels):
        current_labels = []
        current_predictions = []

        for p_i, l_i in zip(pred, label):
            if l_i != -100:
                current_labels.append(id_to_label[l_i])
                current_predictions.append(id_to_label[p_i])

        true_labels.append(current_labels)
        true_predictions.append(current_predictions)
    # return {
    #     "precision": precision_score(true_labels, true_predictions),
    #     "recall": recall_score(true_labels, true_predictions),
    #     "f1": f1_score(true_labels, true_predictions),
    #     "cr": classification_report(true_labels,true_predictions)
    # }
    return {"precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions)}

class WeightedTokenTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        
        #Foward pass of model, use prediction
        outputs = model(**inputs)

        #Get output raw values
        logits = outputs.get("logits") 

        logits = logits.view(-1, logits.shape[-1])       
        labels = labels.view(-1)                          

        if self.class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

#starting models names that will be used for fine tuning
model_names = ["dmis-lab/biosyn-sapbert-bc2gn","microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract","numind/NuNER-v2.0"]
for load_model in model_names:
    tokenizer = AutoTokenizer.from_pretrained(load_model,add_prefix_space=True)

    labels_list = ["O"]

    class_weights = []
    for elem in ["anatomical location","animal","biomedical technique","bacteria","chemical","dietary supplement","DDF","drug","food","gene","human","microbiome","statistical technique"]:
        
        labels_list.append(f"B-{elem}")
        
        labels_list.append(f"I-{elem}")

    label_to_id = {label: i for i, label in enumerate(labels_list)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    #Here the dataset produced with jsonToDataset.py will be loaded
    dataset = load_dataset("json", data_files="dataset-noDevBronze.json", split="train")

    all_labels = list(itertools.chain.from_iterable(dataset["labels"]))

    # CustomWeight logic
    label_counts = Counter(all_labels)
    total = sum(label_counts.values())
    # Add weights dinamcally depending on dataset
    for label in labels_list:
        class_weights.append(total/label_counts[label]) 
    
    dataset = dataset.train_test_split(test_size=0.01, seed=42)

    train_dataset = dataset["train"]
    # # train_dataset = dataset
    val_dataset = dataset["test"]

    train_dataset = train_dataset.map(preprocess)
    val_dataset= val_dataset.map(preprocess)

    model_name = load_model
    
    # Change this values to change number of epochs of training, every entry in list is a new iteration of training starting from the previous 
    epochs = [20,10,10]
    comulative_epoch = 0
    for epoch in epochs:
        comulative_epoch += epoch 
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(labels_list),
            id2label=id_to_label,
            label2id=label_to_id
        )
        print(model)

   
        data_collator = DataCollatorForTokenClassification(tokenizer)

   
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=epoch,
            #per_device_train_batch_size=8,
            eval_steps=3,
            evaluation_strategy="steps",
            # learning_rate=1e-5,  # Più basso per maggiore stabilità
            # per_device_train_batch_size=4,  # Ridotto per evitare OOM
            # per_device_eval_batch_size=4,
            # gradient_accumulation_steps=4,  # Batch effettivo = 16
            # warmup_ratio=0.1,  # 10% dei passi usati per warmup
            # weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=3,
            save_steps=500,
            save_total_limit=2,
            report_to="none",
            # lr_scheduler_type="cosine",  # Più dolce di 'linear'
            # fp16=True, 
        )

        

        trainer = WeightedTokenTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            class_weights=class_weights,
            compute_metrics=compute_metrics
        )

        trainer.train()

        model_name = f"{load_model}-{comulative_epoch}-CW"
        model.save_pretrained(model_name)
        tokenizer.save_pretrained(model_name)

        print(f"\n✅ Training completato. Modello salvato in {model_name}\n")

        # 13. Carica modello salvato e prova inferenza
        print("🔎 Esempio di utilizzo del modello salvato...\n")

