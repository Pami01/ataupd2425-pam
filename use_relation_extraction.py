import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
MODEL_PATH = "./RE-BiomedNLP-2NoRel-5epoch"
LABELS = ["no relation","influence", "is linked to", "target", "located in","impact","change abundance","change effect","used by","affect","part of","interact","produced by","strike","change expression","administered","is a","compared to"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

model = None
tokenizer = None
device = None

def loadModel(MODEL_PATH):
    global model
    global tokenizer
    global device
    print(f"Loading model {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

#Adapted function from relation-extraction_train
def predict_relations(text, entities):
    results_binary = []
    results_ternary_tag = []
    results_ternary_mention = []

    for i, subj in enumerate(entities):
        for j, obj in enumerate(entities):
            if i == j:
                continue

            s_start, s_end = subj["start_idx"], subj["end_idx"]+1
            o_start, o_end = obj["start_idx"], obj["end_idx"]+1
            s_label=subj["label"]
            o_label=obj["label"]
            # Add marker into text
            if s_start < o_start:
                marked_text = f"{text[:s_start]}[E1]{text[s_start:s_end]}[/E1]{text[s_end:o_start]}[E2]{text[o_start:o_end]}[/E2]{text[o_end:]}"
            else:
                marked_text = f"{text[:o_start]}[E2]{text[o_start:o_end]}[/E2]{text[o_end:s_start]}[E1]{text[s_start:s_end]}[/E1]{text[s_end:]}"
                

            inputs = tokenizer(marked_text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()

            #If relation not in labels do not add it inside entries
            relation = id2label[pred]
            if relation == "no relation":
                continue

            entry = {
                "subject_label": s_label,
                "object_label": o_label,
            }
            results_binary.append(entry)

            entry_tag = entry.copy()
            entry_tag["predicate"]= relation
            results_ternary_tag.append(entry_tag)

            entry_mention = entry_tag.copy()
            entry_mention["subject_text_span"] = text[s_start:s_end]
            entry_mention["object_text_span"] = text[o_start:o_end]
            results_ternary_mention.append(entry_mention)


    return {
        "binary_tag_based_relations": results_binary,
        "ternary_tag_based_relations": results_ternary_tag,
        "ternary_mention_based_relations": results_ternary_mention
    }

# TODO Fix main, predict effectively
dump = {}
binary = {}
ternary_tag = {}
ternary_mention = {}

path = r"C:\Users\xLoll\Desktop\GutBrainIE2025\data\Annotations\Dev\json_format\dev.json"


def print_RE(evaluate_path,out_base_path,model_name,id):
    '''
    This function is used to evaluate and json dump the three subtasks of Relation Extraction
    Args:
        evaluate_path: the path to a file that contains title, abstract and the entities in which we will predict the relations
        out_base_path: the base path in which the 3 different json files will be outputted
        model_name: is the model name used in the json files
        id: is the id of the run used in the json files
    Returns:
        None: nothing
    '''
    with open(evaluate_path,"r") as f:
        data = json.load(f) 
    for elem in data.keys():
        print(f"Analizing elem {elem}")
        doc = data[elem]
        metadata = doc["metadata"]
        for loc in ['title','abstract']:

            text = metadata[loc]
            entities = [ent for ent in doc['entities'] if ent["location"] == loc]
            predictions = predict_relations(text,entities)
            binary.update({f"{elem}":{"binary_tag_based_relations":predictions["binary_tag_based_relations"]}})
            ternary_tag.update({f"{elem}":{"ternary_tag_based_relations":predictions["ternary_tag_based_relations"]}})
            ternary_mention.update({f"{elem}":{"ternary_mention_based_relations":predictions["ternary_mention_based_relations"]}})

    os.mkdir(f"{out_base_path}/ataupd2425-pam_T621_{id}_{model_name}")
    with open(f"{out_base_path}/ataupd2425-pam_T621_{id}_{model_name}/ataupd2425-pam_T621_{id}_{model_name}.json", "w") as out:
        json.dump(binary, out, indent=2)
    os.mkdir(f"{out_base_path}/ataupd2425-pam_T622_{id}_{model_name}")
    with open(f"{out_base_path}/ataupd2425-pam_T622_{id}_{model_name}/ataupd2425-pam_T622_{id}_{model_name}.json", "w") as out:
        json.dump(ternary_tag, out, indent=2)
    os.mkdir(f"{out_base_path}/ataupd2425-pam_T623_{id}_{model_name}")
    with open(f"{out_base_path}/ataupd2425-pam_T623_{id}_{model_name}/ataupd2425-pam_T623_{id}_{model_name}.json", "w") as out:
        json.dump(ternary_mention, out, indent=2)
    print("Predictions saved in json files")

if __name__ == "__main__":
    loadModel(MODEL_PATH)
    try:
        print_RE(path,"Test")
    except Exception as e:
        print(f"Something went wrong with file {MODEL_PATH}, {str(e)}")