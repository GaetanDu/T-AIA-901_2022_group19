import typer
import csv
import os
import srsly
import json
from pathlib import Path

import spacy
from spacy.language import Language
from spacy.matcher import Matcher, DependencyMatcher, PhraseMatcher
from spacy.tokens import Token, Span, DocBin
from spacy_language_detection import LanguageDetector
from xgboost import XGBClassifier
import os
import numpy as np


@Language.factory("flag_loc_xgboost", default_config={'path_model':'temp/model.json', 'path_max_length': "temp/max_length.json"})
def flag_destination(nlp, name, path_model, path_max_length):
    return LocXGB(nlp.vocab, path_model, path_max_length)

@Language.factory("language_detector_xgb")
def get_lang_detector(nlp, name):
    return LanguageDetector()

class LocXGB:
    def __init__(self, vocab, path_model, path_max_length): 
        self.model = XGBClassifier()
        self.model.load_model(path_model)
        with open(path_max_length) as f:
            self.max_length = json.load(f)

    def __call__(self, doc):
        # doc = 'je veux aller de marseille à paris'
        # This method is invoked when the component is called on a Doc
        trad = ['DESTINATION', 'DEPARTURE', 'LOC']
        new_ents = []  # Collect the matched spans here
        for idx, ent in enumerate(doc.ents): #('marseille', 'paris')
            if self.max_length['max_length'] - len(doc) > 0:
                example = np.concatenate([[np.mean(ent.vector)], np.pad([np.mean(tok.vector) for tok in doc], (0, self.max_length['max_length'] - len(doc)), 'constant')])
            else:
                example = np.concatenate([[np.mean(ent.vector)], [np.mean(tok.vector) for tok in doc][:self.max_length['max_length']]])
            pred = self.model.predict(example.reshape(1, -1))
            new_ents.append(Span(doc, ent.start, ent.end, label=trad[pred[0]]))
        doc.ents = new_ents
        return doc

    def to_disk(self, path, exclude=tuple()):
        # This will receive the directory path + /my_component
        model_path = path / "model.json"
        try:
            os.mkdir(path)
        except:
            pass
        self.model.save_model(model_path)

    def from_disk(self, path, exclude=tuple()):
        # This will receive the directory path + /my_component
        path_model = path / "model.json"
        self.model = XGBClassifier()
        self.model.load_model(path_model)
        return self

def main(path_data: Path, vectors_model: str, nlp_dir: Path):
    """ Step 1: create the pipeline with a matcher and a language detector."""

    # First: create a simple model from a model
    nlp = spacy.load("fr_core_news_md")
    # Add component to the pipeline
    nlp.add_pipe('language_detector_xgb', last=True)

    examples = []
    labels = []
    doc_bin = DocBin().from_disk(path_data)
    docs = list(doc_bin.get_docs(nlp.vocab))
    max_length = max([len(doc) for doc in docs])
    for doc in docs:
        for ent in doc.ents:
            examples.append(np.concatenate([[np.mean(ent.vector)], np.pad([np.mean(tok.vector) for tok in doc], (0, max_length - len(doc)), 'constant')]))
            if ent.label_ == 'DESTINATION':
                labels.append(0)
            elif ent.label_ == 'DEPARTURE':
                labels.append(1)
            else:
                labels.append(2)
    model = XGBClassifier().fit(examples, labels)
    model.save_model('temp/model.json')
    
    with open("temp/max_length.json", "w") as out_file:
        json.dump({'max_length':max_length}, out_file)

    nlp.add_pipe("flag_loc_xgboost", last=True)
    nlp.to_disk(nlp_dir)

if __name__ == "__main__":
    typer.run(main)
