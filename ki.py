import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Lade den Tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Lade das trainierte Modell
model = load_model('trained_model.h5')
max_len = model.input_shape[1]  # Maximale Sequenzlänge aus dem Modell

# Lade die Labels
with open('labels.pkl', 'rb') as handle:
    y_labels = pickle.load(handle)

print("Modell und Tokenizer erfolgreich geladen!")

# Funktion zum Erzeugen einer Antwort auf Benutzereingaben
def get_response(user_input):
    input_seq = tokenizer.texts_to_sequences([user_input.lower()])
    input_seq = pad_sequences(input_seq, maxlen=max_len, padding='post')
    prediction = model.predict(input_seq)
    predicted_class_index = prediction.argmax(axis=1)[0]

    if predicted_class_index >= len(y_labels):
        return "Entschuldigung, ich habe keine passende Antwort gefunden."

    predicted_response = y_labels[predicted_class_index]
    return predicted_response

# Interaktive Konsole für Benutzereingaben
print("Die KI ist bereit. Gib eine Frage ein (oder 'exit' zum Beenden):")
while True:
    user_input = input("Du: ")
    if user_input.lower() == 'exit':
        print("KI: Auf Wiedersehen!")
        break

    response = get_response(user_input)
    print(f"KI: {response}")