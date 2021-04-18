import pandas as pd
from laserembeddings import Laser
import numpy as np
import os
import time
from IPython import display
from tqdm import tqdm

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

import generate_csv_from_reviews


class SentimentAnalyse(object):
	"""
	SentimentAnalyse generate a model to do sentiment snalyses on sentences
	laserembeddings is used as embedding
	keras is used to build model

	methods:
		generate_model -> train a model from reviews and save it
		load_model -> load a trained model
		model_predict -> take list of sentences and use model to predict sentiment
	"""

	def __init__(self, verbose=1):
		# load laserembeddings models
		# if they're missing, we're downloading them
		try:
			self.laser = Laser()
		except:
			if verbose > 0:
				print("WARNING laserembeddings models missing, downloading ...")
			os.system("python -m laserembeddings download-models")
			self.laser = Laser()

		# load reviews csv
		# if it's missing, we're generating it
		# it is generated with "generate_csv_from_reviews.py" file, who is based on reviews in "sorted_data"
		if not os.path.isfile("labeled_reviews.csv"):
			if verbose > 0:
				print("WARNING csv missing, generating ...")
				start_timer = time.time()
			generate_csv_from_reviews.generate_csv_from_reviews("labeled_reviews.csv")
			if verbose > 0:
				print("time to generate:", round(time.time() - start_timer, 2), "s")

		# load stopwords
		f = open("sorted_data/stopwords", "r")
		self.stopwords = f.read().split("\n")
		self.stopwords.pop(-1)

		self.df_reviews = pd.read_csv("labeled_reviews.csv")

		# initialise model as False so we know he isnt already loaded
		self.model = False

		if verbose > 0:
			print("SentimentAnalyse ready to use.")


	def _train_model(self, model, X, Y, path_model_save="model.h5", verbose=2):
		"""
		params:
			model : keras model -> model to train
			X : list of embedded sentence -> input of model (X)
			Y : list of int, sentiments of X -> output of model (y hat)
			path_model_save : str -> path to save the model
			verbose : int -> show progress if verbose > 0

		return:
			model : keras model -> the trained model
		"""

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
			
		# compile model
		model.compile(
			loss      = tf.keras.losses.BinaryCrossentropy(from_logits=True),
			optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),
			metrics   = [ tf.keras.metrics.BinaryAccuracy() ],
		)
		
		# train model
		model.fit(
			X_train, Y_train, 
			batch_size = 32, 
			epochs     = 1000, 
			validation_split = 0.2,
			callbacks = [
				tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10),
				tf.keras.callbacks.ModelCheckpoint(path_model_save,  monitor='binary_accuracy', mode='max', verbose=0, save_best_only=True)
			],
			verbose=verbose
		)

		# show accuracy
		if verbose > 0:
			print()
			print('Train Accuracy')
			model.evaluate(X_train, Y_train)
			print('Test Accuracy')
			model.evaluate(X_test, Y_test)

		return model


	def preprocess_text_list(self, text_list, lang="en"):
		"""
		return preprocessed text_list
		"""
		# for i in tqdm(range(len(text_list))):
		# 	# strip "\n"
		# 	text_list[i] = text_list[i].strip("\n")
		# 	# lowercase
		# 	text_list[i] = text_list[i].lower()
		# 	# stop words
		# 	text_list[i] = ' '.join([word for word in text_list[i] if not word in self.stopwords])
		print("embedding ...")
		# embedding sentences
		preprocess_text_list = self.laser.embed_sentences(text_list, lang=lang)
		return preprocess_text_list


	def generate_model(self, sample_size=2000, path_model_save="model.h5", verbose=2):
		"""
		params:
			sample_size : int -> number of sentences use to train model
			path_model_save : str -> path to save the model
			verbose : int -> show progress if verbose > 0
		return:
			model : keras model -> trained model
		"""

		# loading data for training
		# the smaller the sample_size is, the faster the model is generated
		df_train = self.df_reviews[self.df_reviews["sentiment"] == 1].head(int(sample_size/2))
		df_train = df_train.append(self.df_reviews[self.df_reviews["sentiment"] == 0].head(sample_size - int(sample_size/2)))
		df_train = df_train.reset_index(drop=True)

		# shuffle DataFrame
		df_train = df_train.sample(frac=1).reset_index(drop=True)

		if verbose > 0:
			print("Train data successfully loaded")
			display.display(df_train.head())
			print(df_train["sentiment"].value_counts())
			print(df_train["rating"].value_counts())
			print("Shape :", df_train.shape)
			print("preprocessing text ...")

		# we're embedding sentences with laserembedding
		# embedded sentence as input X
		# sentiment values as output Y
		X_train = self.preprocess_text_list(df_train["review_text"].values.tolist())
		Y_train = df_train["rating"]

		# min max scale ratins
		Y_train = (Y_train - Y_train.min()) / (Y_train.max() - Y_train.min())

		if verbose > 0:
			print("preprocessing done")
			print("training model ...")

		# creating model
		# every len of embedding sentence are equal to 1024, so for each sentence we have 1024 inputs and 1 output
		model = tf.keras.Sequential([
			tf.keras.Input(shape=(1024,)),
			tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
				tf.keras.layers.BatchNormalization(),
				tf.keras.layers.Dropout(0.25),
			tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
				tf.keras.layers.BatchNormalization(),
				tf.keras.layers.Dropout(0.25),
			tf.keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
				tf.keras.layers.BatchNormalization(),
				tf.keras.layers.Dropout(0.25),
			tf.keras.layers.Dense(8, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
				tf.keras.layers.BatchNormalization(),
				tf.keras.layers.Dropout(0.25),
			tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid),
		])

		# train model
		self.model = self._train_model(model, X_train, Y_train, path_model_save=path_model_save, verbose=verbose)

		# save model
		self.model.save(path_model_save)
		return self.model
		

	def load_model(self, model_path):
		if not os.path.isfile(model_path):
			print(model_path, "is missing")
		else:
			self.model = tf.keras.models.load_model(model_path)
			print(model_path, "loaded")


	def model_predict(self, sentence_list):
		"""
		params:
			sentence : list(str) -> list of sentence
		return:
			if type(sentence_list) == list and self.model != False:
				return predictions
			else:
				return None
		"""
		if type(sentence_list) == list and self.model != False:
			return self.model.predict(self.preprocess_text_list(sentence_list))

		return None


if __name__ == "__main__":

	sentiment_analyser = SentimentAnalyse()

	#sentiment_analyser.load_model("model.h5")

	sentiment_analyser.generate_model(sample_size=10000, path_model_save="model.h5", verbose=2)

	while True:

		ret = input("\nEnter 'exit' to exit\nEnter a sentence:\n")

		if ret == "exit":
			break

		Y = sentiment_analyser.model_predict([ret])

		for output in Y:
			print(output, end=" ")
			if output >= 0.5:
				print("positive")
			else:
				print("negative")

