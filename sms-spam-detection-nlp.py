import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout, LSTM, Bidirectional
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

# Loading the dataset
messages = pd.read_csv("SMSSpamCollection", sep='\t', names=["label", "message"])

# Checking for duplicates
duplicatedRow = messages[messages.duplicated()]

# Splitting ham and spam messages
ham_msg = messages[messages.label == 'ham']
spam_msg = messages[messages.label == 'spam']

# Generate wordclouds
ham_msg_text = " ".join(ham_msg.message.to_numpy().tolist())
spam_msg_text = " ".join(spam_msg.message.to_numpy().tolist())
ham_msg_cloud = WordCloud(width=520, height=260, stopwords=STOPWORDS, max_font_size=50, background_color="black").generate(ham_msg_text)
spam_msg_cloud = WordCloud(width=520, height=260, stopwords=STOPWORDS, max_font_size=50, background_color="black").generate(spam_msg_text)

# Sampling ham messages to match the number of spam messages
ham_msg_df = ham_msg.sample(n=len(spam_msg), random_state=44)
spam_msg_df = spam_msg

# Concatenating ham and spam messages into a single DataFrame
msg_df = pd.concat([ham_msg_df, spam_msg_df], ignore_index=True)

# Adding message length column
msg_df['text_length'] = msg_df['message'].apply(len)

# Calculating average length by label types
labels = msg_df.groupby('label')['text_length'].mean()

# Mapping labels to binary values
msg_df['msg_type'] = msg_df['label'].map({'ham': 0, 'spam': 1})
msg_label = msg_df['msg_type'].values

# Split data into train and test
train_msg, test_msg, train_labels, test_labels = train_test_split(msg_df['message'], msg_label, test_size=0.2, random_state=434)

# Tokenizer setup
max_len = 50
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"
vocab_size = 500
tokenizer = Tokenizer(num_words=vocab_size, char_level=False, oov_token=oov_tok)

# Fit tokenizer on training data
tokenizer.fit_on_texts(train_msg)

# Get the word index
word_index = tokenizer.word_index
print(word_index)
tot_words = len(word_index)
training_sequences = tokenizer.texts_to_sequences(train_msg)
training_padded = pad_sequences(training_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)
testing_sequences = tokenizer.texts_to_sequences(test_msg)
testing_padded = pad_sequences(testing_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

# Dense model architecture
embeding_dim = 16
drop_value = 0.2  # dropout
n_dense = 24
model = Sequential()
model.add(Embedding(vocab_size, embeding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(n_dense, activation='relu'))
model.add(Dropout(drop_value))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fitting a dense spam detector model
num_epochs = 50
early_stop = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(training_padded, train_labels, epochs=num_epochs, validation_data=(testing_padded, test_labels), callbacks=[early_stop], verbose=2)

# Saving and evaluating the Dense model
model.save('Dense_Spam_Detection.h5')
model.evaluate(testing_padded, test_labels)

# Plotting the graphs
def plot_graphs1(var1, var2, string):
    metrics[[var1, var2]].plot()
    plt.title('Training and Validation ' + string)
    plt.xlabel('Number of epochs')
    plt.ylabel(string)
    plt.legend([var1, var2])

metrics = pd.DataFrame(history.history)
metrics.rename(columns={'loss': 'Training_Loss', 'accuracy': 'Training_Accuracy', 'val_loss': 'Validation_Loss', 'val_accuracy': 'Validation_Accuracy'}, inplace=True)
plot_graphs1('Training_Loss', 'Validation_Loss', 'loss')
plot_graphs1('Training_Accuracy', 'Validation_Accuracy', 'accuracy')

# LSTM Spam detection architecture
n_lstm = 20
drop_lstm = 0.2
model1 = Sequential()
model1.add(Embedding(vocab_size, embeding_dim, input_length=max_len))
model1.add(LSTM(n_lstm, dropout=drop_lstm))  # return_sequences=True is giving error due to wrong dimensions
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
num_epochs = 30
early_stop = EarlyStopping(monitor='val_loss', patience=5)
history = model1.fit(training_padded, train_labels, epochs=num_epochs, validation_data=(testing_padded, test_labels), callbacks=[early_stop], verbose=2)

# Saving and evaluating the LSTM model
model1.save('LSTM_Spam_Detection.h5')
metrics = pd.DataFrame(history.history)
metrics.rename(columns={'loss': 'Training_Loss', 'accuracy': 'Training_Accuracy', 'val_loss': 'Validation_Loss', 'val_accuracy': 'Validation_Accuracy'}, inplace=True)
plot_graphs1('Training_Loss', 'Validation_Loss', 'loss')
plot_graphs1('Training_Accuracy', 'Validation_Accuracy', 'accuracy')

# SVM Spam detection - using TF-IDF vectorizer instead of sequences
tfidf_vectorizer = TfidfVectorizer(max_features=500)
train_tfidf = tfidf_vectorizer.fit_transform(train_msg)
test_tfidf = tfidf_vectorizer.transform(test_msg)

svm_model = SVC(C=10, gamma=0.1, kernel='rbf')
svm_model.fit(train_tfidf, train_labels)

# Evaluating SVM model
svm_score = svm_model.score(test_tfidf, test_labels)
print(f"SVM Model Accuracy: {svm_score}")

# BiLSTM Spam detection architecture
model2 = Sequential()
model2.add(Embedding(vocab_size, embeding_dim, input_length=max_len))
model2.add(Bidirectional(LSTM(n_lstm, dropout=drop_lstm)))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training BiLSTM model
num_epochs = 30
early_stop = EarlyStopping(monitor='val_loss', patience=4)
history = model2.fit(training_padded, train_labels, epochs=num_epochs, validation_data=(testing_padded, test_labels), callbacks=[early_stop], verbose=2)

# Saving and evaluating the BiLSTM model
model2.save('BiLSTM_Spam_Detection.h5')
model2.evaluate(testing_padded, test_labels)

# Plotting BiLSTM model
metrics = pd.DataFrame(history.history)
metrics.rename(columns={'loss': 'Training_Loss', 'accuracy': 'Training_Accuracy', 'val_loss': 'Validation_Loss', 'val_accuracy': 'Validation_Accuracy'}, inplace=True)
plot_graphs1('Training_Loss', 'Validation_Loss', 'loss')
plot_graphs1('Training_Accuracy', 'Validation_Accuracy', 'accuracy')
import pickle

# Assuming 'tokenizer' is the tokenizer used during training
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)