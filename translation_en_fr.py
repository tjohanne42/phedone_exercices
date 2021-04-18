import pandas as pd
import os
import numpy as np
from IPython import display
from tqdm import tqdm

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import GRU, TimeDistributed, Dense, Dropout
from keras.losses import sparse_categorical_crossentropy
#from keras.optimizers import adam

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import FrenchStemmer
from nltk.corpus import stopwords


class TranslateEnFr(object):
	"""
	docstring for TranslateEnFr
	"""

	def __init__(self):
		# load text_en_fr
		# if it's missing, we're generating it
		if not os.path.isfile("text_en_fr.csv"):
			self.df = self.generate_csv_from_en_fr_text("text_en_fr.csv")
		else:
			self.df = pd.read_csv("text_en_fr.csv")

		# load stopwords
		f = open("sorted_data/stopwords", "r")
		stopwords_en = f.read().split("\n")
		stopwords_en.pop(-1)

		self.stopwords = stopwords.words('english') + stopwords.words('french') + stopwords_en


		# load lemmatizer for en and fr
		self.lemmatizer_en = WordNetLemmatizer()
		self.stemmer_fr = FrenchStemmer()

		# preprocess text
		self.X, self.Y = self.preprocess(self.df["en"].values.tolist()[:2], self.df["fr"].values.tolist()[:2])

		self.train_model_en_to_fr("model_translate.h5")


	def generate_csv_from_en_fr_text(self, file_path):
		dir_list = os.listdir('Trans1')

		tmp_dict = {"en": [], "fr": []}

		for dir_name in dir_list:
			if ".e" in dir_name:
				file_list = os.listdir('Trans1/' + dir_name)
				f = open('Trans1/' + dir_name + "/" + file_list[0], "r")
				tmp_dict["en"].append(f.read())

			elif ".f" in dir_name:
				file_list = os.listdir('Trans1/' + dir_name)
				f = open('Trans1/' + dir_name + "/" + file_list[0], "r")
				tmp_dict["fr"].append(f.read())
		
		df = pd.DataFrame(data=tmp_dict)

		df.to_csv(file_path, index=False)

		return df


	def tokenize(self, x):
		"""
		tokenize x
		params:
			x : list str -> list of sentences to tokenize
		return:
			tokenized x, tokenizer used
		"""
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(x)
		return tokenizer.texts_to_sequences(x), tokenizer


	def padding(self, x, length=None):
		"""
		padding x
		params:
			x : list str -> list of sentences to padding
		return:
			np array -> padded list
		"""
		return pad_sequences(x, maxlen=length, padding="post")


	def preprocess(self, x, y):
		"""
		params:
			x : list str -> list of en sentences
			y : list str -> list of fr sentences
		return:
			preprocessed x, y
		"""
		for i in tqdm(range(len(x))):
			# strip "\n"
			x[i] = x[i].strip("\n")
			# lowercase
			x[i] = x[i].lower()
			# lemmatize
			x[i] = ' '.join([self.lemmatizer_en.lemmatize(w) for w in x[i].split()])
			# stop words
			x[i] = ' '.join([word for word in x[i].split() if not word in self.stopwords])

		for i in tqdm(range(len(y))):
			# strip "\n"
			y[i] = y[i].strip("\n")
			# lowercase
			y[i] = y[i].lower()
			# lemmatize
			y[i] = ' '.join([self.stemmer_fr.stem(w) for w in y[i].split()])
			# stop words
			y[i] = ' '.join([word for word in y[i].split() if not word in self.stopwords])

		preprocess_x, self.tokenizer_en = self.tokenize(x)
		preprocess_y, self.tokenizer_fr = self.tokenize(y)

		preprocess_x = self.padding(preprocess_x)
		preprocess_y = self.padding(preprocess_y)

		self.max_english_sequence_length = preprocess_x.shape[1]
		self.max_french_sequence_length = preprocess_y.shape[1]
		self.english_vocab_size = len(self.tokenizer_en.word_index)
		self.french_vocab_size = len(self.tokenizer_fr.word_index)

		print('Data Preprocessed')
		print("Max English sentence length:", self.max_english_sequence_length)
		print("Max French sentence length:", self.max_french_sequence_length)
		print("English vocabulary size:", self.english_vocab_size)
		print("French vocabulary size:", self.french_vocab_size)

		return preprocess_x, preprocess_y


	def logits_to_text(self, logits, tokenizer):
	    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
	    index_to_words[0] = '<PAD>'

	    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


	def simple_model(self, input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
	    learning_rate = 0.005
	    
	    model = Sequential()
	    model.add(GRU(256, input_shape=input_shape[1:], return_sequences=True))
	    model.add(TimeDistributed(Dense(1024, activation='relu')))
	    model.add(Dropout(0.5))
	    model.add(TimeDistributed(Dense(self.french_vocab_size, activation='softmax'))) 

	    # Compile model
	    model.compile(loss=sparse_categorical_crossentropy,
	                  #optimizer=adam(learning_rate),
	                  metrics=['accuracy'])
	    return model


	def train_model_en_to_fr(self, path_model_save):
		# Reshaping the input to work with a basic RNN
		# tmp_x = self.padding(self.X, length=self.max_french_sequence_length)
		# tmp_x = tmp_x.reshape((-1, self.Y.shape[-2], 1))

		# tmp_y = self.Y.reshape((-1, tmp_x.shape[-2], 1))

		# print(tmp_y.shape)

		self.X = self.padding(self.X, length=self.max_french_sequence_length)

		self.X = self.X.reshape((-1, self.Y.shape[-2], 1))
		self.Y = self.Y.reshape(self.Y.shape[1], self.Y.shape[0], 1)


		print(self.X.shape)
		print(self.Y.shape)


		# Train the neural network
		simple_rnn_model = self.simple_model(
		    self.X.shape,
		    self.max_french_sequence_length,
		    self.english_vocab_size,
		    self.french_vocab_size)

		print(simple_rnn_model.summary())

		simple_rnn_model.fit(self.X, self.Y, batch_size=1024, epochs=10, validation_split=0.2)


		simple_rnn_model.save(path_model_save)

		self.model = simple_rnn_model

		# Print prediction(s)
		print("Prediction:")
		print(self.logits_to_text(simple_rnn_model.predict(self.X[:1])[0], self.tokenizer_fr))

		print("\nCorrect Translation:")
		print(self.Y[:1])

		print("\nOriginal text:")
		print(self.X[:1])




if __name__ == "__main__":

	translate_en_fr = TranslateEnFr()

