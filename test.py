from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Scegliamo un modello BioBERT già fine-tuned per NER
model_name = "medicalai/ClinicalBERT"

# Caricamento tokenizer e modello
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Creazione della pipeline NER
ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=0  # Raggruppa i token consecutivi con stessa entità
)

# Esempio di testo medico
text = """
The patient was diagnosed with diabetes mellitus and hypertension.
He was prescribed 500mg of Metformin and 10mg of Lisinopril daily.
"""

# Esecuzione NER
entities = ner_pipeline(text)

# Stampa dei risultati
for ent in entities:
    print(f"{ent['word']} ({ent['entity_group']}) [score={ent['score']:.2f}]")
