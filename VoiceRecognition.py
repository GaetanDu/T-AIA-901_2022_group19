import speech_recognition as sr
import langid
import spacy
from spacy import displacy

# Utilisation de Flake8
# Programme qui capture un enregistrement via le microphone/fichier audio
# Et qui traitera les données enregistré


# Fonction de capture de son par le microphone
# Check si la capture est en francais ou pas
# Si oui retourne le contenu de la capture
# Si non vous demande de recommencer l'enregistrement


def capture_mic(microphone):
    r = sr.Recognizer()
    mic = sr.Microphone(device_index=microphone)

    with mic as source:
        r.adjust_for_ambient_noise(source)
        print("you can start to talk")
        audio = r.listen(source)
    text_data = r.recognize_google(audio, language="fr-FR")

    detected = langid.classify(text_data)[0]
    print(detected)
    if detected == "fr":
        print("cest bien du francais")
        print(text_data)
        return text_data
    else:
        print("ce n'est pas du francais veuillez recommencer")
        capture_mic(1)


text_to_process = capture_mic(1)


def Extract_Data(Source):
    nlp_instance = spacy.load("fr_core_news_md")
    load_data = nlp_instance(Source)

    displacy.render(load_data, style="ent")
    displacy.render(load_data, style="dep")

    return 0


Extract_Data(text_to_process)
