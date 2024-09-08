import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
import pickle

# Schritt 1: Lade den Datensatz
data = pd.read_csv('dataset.csv')

# Schritt 2: Erstelle und trainiere den Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['input'])

with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)



# Schritt 3: Texte in numerische Sequenzen umwandeln
X = tokenizer.texts_to_sequences(data['input'])
max_len = max(len(x) for x in X)
X = pad_sequences(X, maxlen=max_len, padding='post')
y, y_labels = pd.factorize(data['output'])

# Schritt 4: Erstelle das Modell
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1,
              output_dim=100,
              input_length=max_len,
              trainable=True),
    LSTM(512, activation='tanh', return_sequences=False, kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(len(y_labels), activation='softmax', kernel_regularizer=l2(0.01))
])

with open('labels.pkl', 'wb') as handle:
    pickle.dump(y_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Schritt 5: Kompiliere das Modell
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Schritt 6: Modell Checkpoints einrichten
checkpoint_filepath = 'model_checkpoint_lstm.weights.h5'
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
                                      save_weights_only=True,
                                      monitor='val_accuracy',
                                      mode='max',
                                      save_best_only=True,
                                      verbose=1)

# Schritt 7: Trainiere das Modell
model.fit(X, y, epochs=10000, validation_split=0.2, callbacks=[checkpoint_callback])

# Speichere das Modell und die Tokenizer-Datei
model.save('trained_model.h5')
print("Training abgeschlossen und Modell gespeichert!")