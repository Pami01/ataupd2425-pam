import json

entity_labels = ["anatomical location","animal","biomedical technique","bacteria","chemical","dietary dupplement","DDF","drug","food","gene","human","microbiome","statistical technique"]

class Document:
    def __init__(self,id,title,abstract,year,journal,authors):
        self._id = id
        self._title = title
        self._abstract = abstract
        self._year= year
        self._journal = journal
        self._authors = authors
        self._entities = []

    def __str__(self):

        returned=f"{self._id},{self._title},entities:"
        for entity in self._entities:
            returned += str(entity)
            returned += "\n"
        return returned
    
    def add_entity(self,entity):
        if type(entity) != Entity:
            raise Exception("Not an entity")
        self._entities.append(entity)
    
class Entity:
    locations = ["title","abstract"]

    def __init__(self,start_idx,end_idx,location,text_span,label):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.location = location
        self.text_span = text_span
        self.label= label
    
    @property
    def start_idx(self):
        return self._start_idx
    
    @property
    def location(self):
        return self._location
    
    @property
    def label(self):
        return self._label

    @start_idx.setter
    def start_idx(self, start_idx):
        if start_idx < 0:
            raise Exception("Index negative")
        self._start_idx = start_idx

    @location.setter
    def location(self,location):
        if location not in self.locations:
            raise Exception("No location")
        self._location = location
    
    @label.setter
    def label(self,label):
        if label not in entity_labels:
            raise Exception("Not in entity_labels")
        self._label = label
    def __str__(self):
        return f"{self.text_span},{self.label}"
    
class Parser:
    obj = None
    def __init__(self):
        pass
    def decode_doc(self,filepath):
        with open(filepath) as f:
            self.obj=json.load(f)
        # Analyzing parsed object
        for key in self.obj.keys():
            # Parsing every doc in json file
            doc = self.obj[key]
            for key in doc.keys():
                # Every bit of information about every single document
                metadata = doc['metadata']
                entities = doc['entites']
                relations = doc['relations']
                
                
    
    def __str__(self):
        return str(self.obj)

d=Document(1,"title bro","abstract",1999,"Ciao","Ciao")
entity= Entity(1,2,"title","ciao","DDF")
d.add_entity(entity)
print(d)

p = Parser()
p.decode_doc("test.json")



