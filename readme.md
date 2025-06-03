# GutBrainIE Baselines & Evaluation

This repository provides the complete pipeline for my submitted solution for the [GutBrainIE](https://hereditary.dei.unipd.it/challenges/gutbrainie/2025/) challenge – the sixth task of the [BioASQ](https://www.bioasq.org/) Lab @ [CLEF 25](https://clef2025.clef-initiative.eu/).

---

I created an entire logic workflow sorrounding CRF that can be found in Logic.py, some of this logic is used also in NER prediction of fine tuned models.

# CRF models
## Training 

The Conditional Random Field model could be trained using the file main.py, changing the inside _load_ (true) variable we can start the training of the CRF model

## Usage
Using the same main.py file, changing the inside _load_ (false) variable we can start the prediction of the CRF model


# Bert models
Before executing the pipeline to train BERT models for NER  you have to create a specific format to work with this data, to obtain this you have to run jsonToDataset.py
## Training NER
Using trainModel.py you can train a NER models starting from different types of models found on HuggingFace

## Using NER 
Using useModel.py you can infer entities starting from pre-trained models saved using trainModel.py.

## Training RE
To train Relation Extraction models you have to use relation_extraction_train.py, 

## Using RE models
To use Relation Extraction models you have to use use_relation_extraction.py

You can also use useModel.py to work with multiple models of NER, produce RE compatible files and sequentially work with RE models.