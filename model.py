from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from anago.layers import CRF

import numpy as np
import os

# Constants

max_features = 20000
maxlen = 100
MAX_WORDS = 20000

# Labels-index dictionary
labels_index = {}
labels_index['O'] = 0
labels_index['B-ORG'] = 1
labels_index['I-ORG'] = 2
labels_index['B-PER'] = 3
labels_index['I-PER'] = 4
labels_index['B-LOC'] = 5
labels_index['I-LOC'] = 6
labels_index['B-MISC'] = 7
labels_index['I-MISC'] = 8

def load_data(file_path):
	# Complete data (2D arrays)
	words = []
	tags = []

	# Tmp data
	tmp_words = ""
	#tmp_tags = []
	for line in open(file_path, encoding='utf-8'):
		line = line.strip()

		# Check for newline (sentece separator)
		if not line: # Prepare to read new sentence
			# Append non-empty words and tags
			if tmp_words:
				words.append(tmp_words[1:])
				#tags.append(tmp_tags)
			# Clear tmp arrays
			tmp_words = ""
			#tmp_tags = []
		else: # Currently reading sentence
			word, _, _, tag = line.split()
			
			tmp_words += " " + word
			#tmp_tags.append(labels_index[tag])
			tags.append(labels_index[tag])

	return words, tags


class BidirLSTMCRF:
	""" 
	Implementation of Bidirectional LSTM-CRF neural network for sequence labeling using Keras.

    References:
    https://arxiv.org/abs/1603.01360
    """

	def __init__(self,
				 labels_cnt,
				 dropout=0.5,
				 embeddings=None,
				 use_crf=True,
				 optimizer='adam'):
		"""
		Creates Bidirectional LSTM-CRF model.

		Args:
		    labels_cnt (int): count of entity labels.
		    dropout (float): dropout rate.
		    embeddings (numpy array): word embedding matrix.
		    use_crf (boolean): use CRF as last layer.
		"""
		self.labels_cnt = labels_cnt
		self.dropout = dropout
		self.embeddings = embeddings
		self.use_crf = use_crf
		self.optimizer = optimizer
		self.word_lstm_size=100
		self.fc_dim=100

		# Build model
		self.model, loss = self.build_model()
		self.model.compile(self.optimizer, loss, metrics=['accuracy'])

	def build_model(self):
		"""
		Builds Bidirectional LSTM-CRF model.
		"""

		# Word embedding - TODO
		word_embeddings = Dropout(self.dropout)

		# Model
		model = Sequential()
		# TODO - Embeddings
		model.add(Embedding(max_features, self.word_lstm_size * 2, input_length=maxlen))

		# Bidirectional LSTM
		#bi_lstm = Bidirectional(LSTM(units=self.word_lstm_size, return_sequences=True))
		model.add(Bidirectional(LSTM(units=self.word_lstm_size, return_sequences=True)))
		#bi_lstm = Dense(self.fc_dim, activation='tanh')(bi_lstm)
		model.add(Dense(self.fc_dim, activation='tanh'))
		#model.add(bi_lstm)
		model.add(Dropout(self.dropout)) # TODO

		# CRF
		if self.use_crf:
			crf = CRF(self.labels_cnt, sparse_target=False)
			loss = crf.loss_function
			model.add(crf)
		else:
			loss = 'categorical_crossentropy'
			model.add(Dense(self.labels_cnt, activation='softmax'))

		return model, loss

	def train(self, x_train, y_train, x_valid=None, y_valid=None, shuffle=True, epochs=1, batch_size=1):
		# Check for validation data
		if x_valid and y_valid:
			validation_data = [x_valid, y_valid]
		else:
			validation_data = None

		# Train model
		self.model.fit(x_train, 
					   y_train, 
					   batch_size=batch_size, 
					   epochs=epochs,
					   validation_data=validation_data,
					   shuffle=shuffle)

	def predict(self, x, batch_size=None, verbose=0, steps=None):
		print('Predicting for ' + str(len(x)) + ' inputs...')
		predictions = predict(x, batch_size=batch_size, verbose=verbose, steps=steps)
		return predictions

# Load CoNNL 2003 dataset
print('Loading data...')
train_words, train_tags = load_data('../../ML-Internet-DataSets/conll2003/en/train.txt')#load_data_and_labels(train_path)
validation_words, validation_tags = load_data('../../ML-Internet-DataSets/conll2003/en/valid.txt')#load_data_and_labels(valid_path)
test_words, test_tags = load_data('../../ML-Internet-DataSets/conll2003/en/test.txt')#load_data_and_labels(test_path)

# Tokenizer
tokenizer = Tokenizer(nb_words=MAX_WORDS)
tokenizer.fit_on_texts(train_words)
train_data = tokenizer.texts_to_sequences(train_words)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

train_data = sequence.pad_sequences(train_data, maxlen=maxlen)
train_tags = to_categorical(np.asarray(train_tags))

#train_words = sequence.pad_sequences(train_words, maxlen=maxlen)
#validation_words = sequence.pad_sequences(validation_words, maxlen=maxlen)

lstm_model = BidirLSTMCRF(5)
lstm_model.train(train_data, train_tags)



