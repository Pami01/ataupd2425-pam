from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import load_dataset
import torch
import torch.nn as nn
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import torch
import torch.nn as nn
from collections import Counter
import itertools

class WeightedTokenTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")  # [batch, seq_len, num_classes]

        # Flatten logits and labels
        logits = logits.view(-1, logits.shape[-1])        # [batch * seq_len, num_classes]
        labels = labels.view(-1)                          # [batch * seq_len]

        if self.class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss



# 1. Imposta modello e tokenizer
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# 2. Label Mapping
labels_list = ["O"]

#labels_list = ["O", "B-Drug", "I-Drug", "B-Disease", "I-Disease", "B-Gene"]
class_weights = []
for elem in ["anatomical location","animal","biomedical technique","bacteria","chemical","dietary supplement","DDF","drug","food","gene","human","microbiome","statistical technique"]:
    
    labels_list.append(f"B-{elem}")
    
    labels_list.append(f"I-{elem}")

label_to_id = {label: i for i, label in enumerate(labels_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

# 3. Carica dataset JSON
dataset = load_dataset("json", data_files="dataset-complete.json", split="train")


# Flatten tutte le etichette token-level in un'unica lista
all_labels = list(itertools.chain.from_iterable(dataset["labels"]))

# Conta le frequenze
label_counts = Counter(all_labels)
total = sum(label_counts.values())
# Add weights dinamcally depending on dataset
for label in labels_list:
    class_weights.append(total/label_counts[label]) 
# Split automatico 80% train, 20% validation
# dataset = dataset.train_test_split(test_size=0.01, seed=42)

# train_dataset = dataset["train"]
# # train_dataset = dataset
# val_dataset = dataset["test"]

# 4. Funzione di preprocessamento
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

# 5. Applica preprocessamento
train_dataset = dataset.map(preprocess)


epochs = [10,2,2]
comulative_epoch = 0
for epoch in epochs:
    comulative_epoch += epoch 
    # 6. Prepara modello
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(labels_list),
        id2label=id_to_label,
        label2id=label_to_id
    )

    # 7. Data Collator per il padding automatico delle labels
    data_collator = DataCollatorForTokenClassification(tokenizer)

# 8. Parametri di training
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epoch,
        per_device_train_batch_size=8,
        logging_dir="./logs",
        logging_steps=1,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
    )

    
    # 9. Trainer Huggingface
    trainer = WeightedTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        class_weights=class_weights
    )

    # 10. Inizia il training
    trainer.train()

    model_name = f"BiomedNLP-PubMedBERT-base-uncased-abstract-{comulative_epoch}-CW-xtreme"
    model.save_pretrained(model_name)
    tokenizer.save_pretrained(model_name)

    print(f"\n✅ Training completato. Modello salvato in {model_name}\n")

    # 13. Carica modello salvato e prova inferenza
    print("🔎 Esempio di utilizzo del modello salvato...\n")

