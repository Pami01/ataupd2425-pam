import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

MODEL_PATH = "./re_output/checkpoint-210"
LABELS = ["influence", "is linked to", "target", "located in","impact","change abundance","change effect","used by","affect","part of","interact","produced by","strike","change expression","administered","is a","compared to"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

#Adapted function from relation-extraction_train
def predict_relations(text, entities):
    results_binary = []
    results_ternary_tag = []
    results_ternary_mention = []

    for i, subj in enumerate(entities):
        for j, obj in enumerate(entities):
            if i == j:
                continue

            s_start, s_end = subj["start"], subj["end"]
            o_start, o_end = obj["start"], obj["end"]

            # Add marker into text
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

            inputs = tokenizer(marked_text, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                outputs = model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()

            #If relation not in labels do not add it inside entries
            relation = id2label[pred]
            if relation == "no_relation":
                continue

            entry = {
                "subject": subj["id"],
                "object": obj["id"],
                "predicate": relation
            }
            results_binary.append(entry)

            entry_tag = entry.copy()
            entry_tag["subject_semantic_tag"] = subj["type"]
            entry_tag["object_semantic_tag"] = obj["type"]
            results_ternary_tag.append(entry_tag)

            entry_mention = entry_tag.copy()
            entry_mention["subject_mention_text"] = text[s_start:s_end]
            entry_mention["object_mention_text"] = text[o_start:o_end]
            results_ternary_mention.append(entry_mention)

    return {
        "binary_tag_based_relations": results_binary,
        "ternary_tag_based_relations": results_ternary_tag,
        "ternary_mention_based_relations": results_ternary_mention
    }

# TODO Fix main, predict effectively
if __name__ == "__main__":
    with open("test_doc.json") as f:
        data = json.load(f) 
    for elem in data.keys():
        data[elem]["metadata"]['abstract']
    result = predict_relations(data["text"], data["entities"])

    with open("predictions.json", "w") as out:
        json.dump(result, out, indent=2)

    print("Predizioni salvate in predictions.json")
