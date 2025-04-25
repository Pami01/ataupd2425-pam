from transformers import pipeline

ner_pipeline = pipeline("ner", model="./biobert-ner-finetuned", tokenizer="./biobert-ner-finetuned", aggregation_strategy="max")

text = "Aspirin is used to treat fever and inflammation at Microsoft."

results = ner_pipeline(text)

for entity in results:
    print(f"{entity['word']} -> {entity['entity_group']} (score: {entity['score']:.2f})")
