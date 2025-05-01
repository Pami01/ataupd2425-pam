import json
from Logic import Parser


BASE_DOC_PATHS = [r"data\Annotations\Train\bronze_quality\json_format\train_bronze.json",r"data\Annotations\Dev\json_format\dev.json",r"data\Annotations\Train\gold_quality\json_format\train_gold.json",r"data\Annotations\Train\platinum_quality\json_format\train_platinum.json",r"data\Annotations\Train\silver_quality\json_format\train_silver.json"]
p = Parser()
for path in BASE_DOC_PATHS:
    p.decode_doc(path)

dataset = []
labels = ["anatomical location","animal","biomedical technique","bacteria","chemical","dietary supplement","DDF","drug","food","gene","human","microbiome","statistical technique"]
numbers = {}
for doc in p.docs:
    tokens =[]
    labels =[]
    for elem in doc.ner:
        tokens.append(elem[0])
        if elem[2] not in numbers:
            numbers.update({elem[2]:0})
        numbers[elem[2]]+=1
        labels.append(elem[2])
    obj = {"tokens":tokens,"labels":labels}
    dataset.append(obj)
with open("dataset-complete.json","w") as f:
    json.dump(dataset,f)
print(numbers)