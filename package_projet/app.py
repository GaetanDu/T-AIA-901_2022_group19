import spacy
import os
os.chdir("package_projet/")
from recognizer.Recognizer import RecognizerRequest
from pathfinding.pathfinding import getBestPath
os.chdir("localizer_model/")
from localizer_model.scripts import create_matcher_model

def get_dest_and_dep(phrase):
    nlp = spacy.load("temp/nlp_regex")
    doc = nlp(phrase)
    results = [ent.text for ent in doc.ents if ent.label_=='DESTINATION' or ent.label_=='DEPARTURE']
    if len(results) == 2:
        return results
    return []

if __name__ == "__main__":

    phrase = RecognizerRequest()
    departure, arrival = get_dest_and_dep(phrase)
    bestTrips = getBestPath([departure, arrival])
    for i in range(len(bestTrips)):
        if bestTrips[i].path is not None:
            print(f"Test Numero : {i+1} - {bestTrips[i]}")
        else:
            if bestTrips[i].startTrainId is None or bestTrips[i].EndTrainId is None:
                print(f"Test Numero : {i+1} - Pas pu trouver de chemin")
            else:
                print(f"Test Numero : {i+1} - Pas pu trouver de chemin")