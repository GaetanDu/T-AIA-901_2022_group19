import typer
import pandas as pd
import json
from collections import Counter
from pathlib import Path
import spacy
from spacy.tokens import DocBin, Span


def main(perc_training: float, csv_loc: Path, nlp_dir: Path, train_corpus: Path, test_corpus: Path):
    """ Step 2: Once we have done the manual annotations, create corpora in spaCy format. """
    nlp = spacy.load(nlp_dir, exclude="flag_loc, language_detector")
    docs = []
    gold_ids = []
    data = pd.read_csv(csv_loc, encoding='latin3')
    for idx, example in data.iterrows():
        sentence = example["TEXT"]
        doc = nlp(sentence)
        new_ents = []
        for ent in doc.ents:
            if ent.text in example['DESTINATION']:
                new_ents.append(Span(doc, ent.start, ent.end, label='DESTINATION'))
            elif ent.text in example['DEPARTURE']:
                new_ents.append(Span(doc, ent.start, ent.end, label='DEPARTURE'))
            elif ent.text in str(example['STEP']):
                new_ents.append(Span(doc, ent.start, ent.end, label='STEP'))
            else:
                new_ents.append(ent)
        gold_ids.append(idx)
        doc.ents = new_ents
        docs.append(doc)

    train_docs = DocBin()
    test_docs = DocBin()
    length = len(gold_ids)
    for index in gold_ids[int(perc_training*length):]:
        train_docs.add(docs[index])
    for index in gold_ids[:length - int(perc_training*length)]:
        test_docs.add(docs[index])

    train_docs.to_disk(train_corpus)
    test_docs.to_disk(test_corpus)


if __name__ == "__main__":
    typer.run(main)
