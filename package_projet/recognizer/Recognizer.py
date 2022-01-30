import speech_recognition as sr

def RecognizerRequest():
    try:
        MyRequest = sr.Recognizer()
        with sr.Microphone() as source:
            MyRequest.adjust_for_ambient_noise(source, duration=0.2)

            print("Vous Pouvez parler")
            audio = MyRequest.listen(source)
            userRequest = MyRequest.recognize_google(audio, language="fr-FR")
            print(f"Votre phrase: {userRequest}")
            return userRequest

    except sr.RequestError as error:
        print(f"error : {error}")
        return 1
