from functools import partial
from pathlib import Path
from typing import Iterable, Callable
import spacy
from spacy.training import Example
from spacy.tokens import DocBin


@spacy.registry.readers("MyCorpus.v1")
def create_docbin_reader(file: Path) -> Callable[["Language"], Iterable[Example]]:
    return partial(read_files, file)


def read_files(file: Path, nlp: "Language") -> Iterable[Example]:
    # we run the full pipeline and not just nlp.make_doc to ensure we have entities and sentences
    # which are needed during training of the entity linker
    with nlp.select_pipes(disable="entity_linker"):
        doc_bin = DocBin().from_disk(file)
        docs = doc_bin.get_docs(nlp.vocab)
        for doc in docs:
            yield Example(nlp(doc.text), doc)

  - name: evaluate
    help: "Final evaluation on the dev data and printing the results"
    script:
      - "python ./scripts/evaluate.py ./training/model-best/ corpus/${vars.dev}.spacy"
    deps:
      - "training/model-best"
      - "corpus/${vars.dev}.spacy"

  # These are additional custom commands that are not run as part of the main
  # "run" list. They can depend on third-party libraries etc. Here are just
  # some examples of what's possible.
  - name: setup
    help: Install dependencies
    script:
      - "python -m pip install -r requirements.txt"
    deps:
      - "requirements.txt"

  - name: clean
    help: "Remove intermediate files"
    script:
      - "rm -rf training/*"
      - "rm -rf corpus/*"
      - "rm -rf temp/${vars.kb}"
      - "rm -rf temp/${vars.nlp}"
