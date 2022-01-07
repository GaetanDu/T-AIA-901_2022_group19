import typer
import csv
import os
import srsly
from pathlib import Path

import spacy
from spacy.kb import KnowledgeBase
from spacy.language import Language
from spacy.matcher import Matcher, DependencyMatcher, PhraseMatcher
from spacy.tokens import Token, Span
from spacy_language_detection import LanguageDetector


@Language.factory("flag_loc", default_config={'path_departure':'./assets/patterns_departure.jsonl',
                                              "path_destination": './assets/patterns_destination.jsonl',
                                             "path_steps": './assets/patterns_steps.jsonl'})
def flag_destination(nlp, name, path_departure, path_steps, path_destination):
    return LocMatcher(nlp.vocab, path_departure, path_steps, path_destination)

@Language.factory("language_detector")
def get_lang_detector(nlp, name):
    return LanguageDetector()

class LocMatcher:
    def __init__(self, vocab, path_departure, path_steps, path_destination): 
        self.matcher = Matcher(vocab)
        self.patterns_departure = [[pattern for pattern in srsly.read_jsonl(path_departure)]]
        self.patterns_steps = [[pattern for pattern in srsly.read_jsonl(path_steps)]]
        self.patterns_destination = [[pattern for pattern in srsly.read_jsonl(path_destination)]]
        self.matcher.add('DEPARTURE', self.patterns_departure)
        self.matcher.add('STEPS', self.patterns_steps)
        self.matcher.add('DESTINATION', self.patterns_destination)
    def __call__(self, doc):
        # This method is invoked when the component is called on a Doc
        new_ents = []  # Collect the matched spans here
        for idx, ent in enumerate(doc.ents):
            previous_span = doc[ent.start-2: ent.start]
            match = self.matcher(previous_span)
            if match: # ensure that there are more than first term 'à' or 'de'
                new_ents.append(Span(doc, ent.start, ent.end, label=match[0][0]))
            else:
                new_ents.append(ent)
        doc.ents = new_ents
        return doc
    def to_disk(self, path, exclude=tuple()):
        # This will receive the directory path + /my_component
        data_path = path / "data.json"
        with data_path.open("w", encoding="utf8") as f:
            f.write(json.dumps({'A':'3'}))

    def from_disk(self, path, exclude=tuple()):
        # This will receive the directory path + /my_component
        a=1
        return self

def main(path_departure: Path, path_destination: Path, path_steps: Path, vectors_model: str, nlp_dir: Path):
    """ Step 1: create the pipeline with a matcher and a language detector."""

    # First: create a simple model from a model
    nlp = spacy.load("fr_core_news_md")
    nlp.add_pipe('language_detector', last=True)
    nlp.add_pipe("flag_loc", last=True)  # Add component to the pipeline
    nlp.to_disk(nlp_dir)


if __name__ == "__main__":
    typer.run(main)
