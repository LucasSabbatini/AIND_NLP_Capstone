"""

ASSUMPTIONS:

	imported text is already preprocessed, that being ponctuation romoved
"""


import tensorflow as tf 
from collections import Counter
## data

with open('data/small_vocab_en', 'r') as f:
	english_sentences = f.read()
	f.close()
with open('data/small_vocab_fr', 'rt', encoding='utf-8') as f:
	french_sentences = f.read()
	f.close()


# PREPROCESSING

def create_dictionaries(word_set):
	"""
	ARGUMENT:
	data : list of sentences (strings)

	RETURN: ( vocab_to_id, id_to_vocab )
	vocab_to_id : 
	id_t0_vocab : 

	NOTES:
		create '<GO>', '<UNK>', '<EOS>' and '<PAD>' entries in the dictionaries

	"""
	words_counter = Counter([word for sentence in data for word in sentece.split()])
	words_list = [word for word, _ in word_counter.most_common()]

	# Adding tokens to words list
	tokens_to_add = ['<PAD>', '<GO>', '<EOS>']
	words_list = tokens_to_add + words_list

	# dictionaries
	word_to_id = {word: idx for idx, word in enumerate(words_list)}
	id_to_word = {idx: word for word, idx in words_to _id.items()}

	return word_to_id, id_to_word


def pad(sequences, max_length=None):
	"""
	ARGUMENTS:
	sequences : list of sequences (list of word ids)

	RETURNS:
	padded_sequences : list with padded sequences

	"""
	# if max length is not defined, set length to the longest sequence,
	# then return the padded sequences
	if max_length is None:
		length = max([len(sequence) for sequence in sequences])
		return [sequence + [0]*(length-len(sequence)) for sequence in sequences]
	
	# else, 
	padded_sequences = []
	for sequence in sequences:
		if len(sequence) >= max_length:
			padded_sequences.append(sequence[:max_length])
		else:
			padded_sequences.append(sequence+[0]*(max_length-len(sequence)))
	return padded_sequences

def tokenize(data):
	"""


	ARGUMENT:
	data : list of sentences (strings)	


	RETURN:

	NOTES:
		Add '<GO>' token to the beginning o each target sentence and '<EOS>' 
		at the end.

	"""

	# GETTING DICTIONARIES
	word_to_id, id_to_word = create_dictionaries(data)

	# CONVERTING SENTENCES TO SEQUENCE OF IDS
	sentences = [[word_to_id[word] for word in sentence.split()] for sentence in data]
	# sentences = [word_to_id[word] for sentence in data for word in sentece.aplit()]
	# for sentence in data:
	# 	sentences.append([word_to_id[word] for word in sentence.split()])

	return sentences, word_to_id, id_to_word

def preprocess(source_date, target_data):
	"""
	
	ARGUMETS:
	source_data : list of sentences (strings) in source language
	target_data : list of sentences (strings) in target language

	"""
	# source_vocab_to_id, source_id_to_vocab = get_vocabs(source_data)
	# target_vocab_to_id, target_id_to_vocab = get_vocabs(target_data)

	tokenized_source, source_word_to_id, source_id_to word = tokenize(source_data, source_vocab_to_id)
	tokenized_target, target_word_to_id, target_id_to_word = tokenize(target_data, target_vocab_to_id)

	# ADD <EOS> AT THE END AND <GO> AT THE BEGINING OF EACH TERGET SENTENCE
	for i in range(len(tokenized_target)):
		tokenized_target[i] = [target_word_to_id['<GO>']] + tokenized_target[i] + [target_word_to_id['<EOS>']]

	# PAD
	preproc_source = pad(tokenized_source)
	preproc_target = pad(tokenized_target)

	return preproc_source, preproc_target, source_word_to_id, target_word_to_id


# OBJECT ORIENTED


import tensorflow as tf 
from collections import Counter
## data

with open('data/small_vocab_en', 'r') as f:
	english_sentences = f.read()
	f.close()
with open('data/small_vocab_fr', 'rt', encoding='utf-8') as f:
	french_sentences = f.read()
	f.close()



class Tokenizer(object):
	def __init__(self, data):
		create_dictionaries(self, data) 

	def create_dictionaries(self, data):
		"""
		ARGUMENT:
		data : list of sentences (strings)

		RETURN: ( vocab_to_id, id_to_vocab )
		vocab_to_id : 
		id_t0_vocab : 

		NOTES:
			create '<GO>', '<UNK>', '<EOS>' and '<PAD>' entries in the dictionaries

		"""
		words_counter = Counter([word for sentence in data for word in sentece.split()])
		words_list = [word for word, _ in word_counter.most_common()]

		# Adding tokens to words list
		tokens_to_add = ['<PAD>', '<GO>', '<EOS>']
		words_list = tokens_to_add + words_list

		# dictionaries
		self.word_to_id = {word: idx for idx, word in enumerate(words_list)}
		self.id_to_word = {idx: word for word, idx in words_to _id.items()}
		self.vocab_len = len(words_list)

	def tokenize(self, data):
		"""

		"""

		# CONVERTING SENTENCES TO SEQUENCE OF IDS
		return [[self.word_to_id[word] for word in sentence.split()] for sentence in data]
		
def pad(sequences, pad_int, max_length=None):
	"""
	ARGUMENTS:
	sequences : list of sequences (list of word ids)

	RETURNS:
	padded_sequences : list with padded sequences

	"""
	# if max length is not defined, set length to the longest sequence,
	# then return the padded sequences
	if max_length is None:
		length = max([len(sequence) for sequence in sequences])
		return [sequence + [pad_int]*(length-len(sequence)) for sequence in sequences]
	
	# else, 
	padded_sequences = []
	for sequence in sequences:
		if len(sequence) >= max_length:
			padded_sequences.append(sequence[:max_length])
		else:
			padded_sequences.append(sequence+[pad_int]*(max_length-len(sequence)))
	return padded_sequences

def preprocess(source_date, target_data):
	"""
	
	ARGUMETS:
	source_data : list of sentences (strings) in source language
	target_data : list of sentences (strings) in target language

	"""
	# source_vocab_to_id, source_id_to_vocab = get_vocabs(source_data)
	# target_vocab_to_id, target_id_to_vocab = get_vocabs(target_data)


	source_tokenizer = Tokenizer(source_data)
	target_tokenizer = Tokenizer(target_data)

	tokenized_source = source_tokenizer.tokenize(source_data)
	tokenized_target = target_tokenizer.tokenize(target_data)


	# ADD <EOS> AT THE END AND <GO> AT THE BEGINING OF EACH TERGET SENTENCE
	for sequence in tokenized_target:
		sequence.append(target_word_to_id['<EOS>'])

	# PAD
	preproc_source = pad(tokenized_source, source_tokenizer.word_to_id['<PAD>'])
	preproc_target = pad(tokenized_target, target_tokenizer.word_to_id['<PAD>'])


	return preproc_source, preproc_target, source_tokenizer, target_tokenizer


preproc_english_sentences, preproc_french_sentences, engish_tokenizer, french_tokenizer = preprocess(english_sentences, french_sentences)

def preprocess_decoder_input(target_data, word_to_id):
	"""
	Remove last item and insert <GO> before each sequence
	"""
	return [[word_to_id['<GO>']] + sequence[:-1] for sequence in target_data]

tmp_french_sequences = preprocess_decoder_input(preproc_french_sentences, french_tokenizer.word_to_id)



# TENSORFLOW

# MODEL INPUTS

def model_inputs():
	"""
	Create tf placeholders for inputs, targets, learning_rate, and lengths of source
	and target sequences.

	"""

	inputs = tf.placeholders(tf.int32, [None, None], name='input')

	targets = tf.placeholder(tf.float32, [None, None], name='target')

	learning_rate = tf.placeholder(tf.float32)

	keep_prob = tf.placeholder(tf.float32, name='keep_prob')

	target_sequence_length = tf.placeholder(tf.int32, [None,], name='target_sequence_length')

	max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_length')

	source_sequence_length = tf.placeholder(tf.int32, name='source_sequence_length')

	return (inputs,
			targets,
			learning_rate,
			keep_prob,
			target_sequence_length,
			max_target_sequence_length,
			source_sequence_length)


def process_decoder_input(target_data, word_to_id, batch_size):
	"""
	Removing last item and inserting <GO> 

	"""

	processed_targets = tf.stride_slice(target_data, [0,0], [batch_size, -1], [1,1])
	processed_targets = tf.concat([tf.fill([batch_size, 1], word_to_id['<GO>']), processed_targets],1)

	return processed_targets


# ENCODING

def encoding_layer(rnn_inputs,
					rnn_size, 
					num_layers,
					keep_prob,
					source_sequence_length,
					source_vocab_size,
					encoding_embedding_size):
	encoder_embed_input = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, encoding_embedding_size)

	encoder_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(rnn_size), output_keep_prob=keep_prob) for _ in range(num_layers)])

	encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
													  encoder_embed_input,
													  sequence_length-source_sequence_length,
													  dtype=tf.float32)
	return encoder_output, encoder_state

# DECODING TRAINING LAYER

def decoding_layer_train(encoder_state,
						 dec_cell,
						 dec_embed_input,
						 target_sequence_length,
						 max_summary_length,
						 output_layer,
						 keep_prob):
	
	training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
														sequence_length=target_sequence_length,
														time_major=False)

	training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
													   training_helper,
													   encoder_state,
													   output_layer)

	training_decoder_output = tf.contrib.seq2seq.dynamic_decode(training_decoder,
																impute_finished=True,
																maximum_iterations=max_summary_lenght)[0]

	return training_decoder_output

# DECODING INFERENCE LAYER

def decoding_layer_infer(encoder_state,
						 dec_cell,
						 dec_embeddings,
						 start_of_sequence_id,
						 end_of_sequence_id,
						 max_target_sequence_length,
						 vocab_size,
						 output_layer,
						 batch_size,
						 keep_prob):
	
	start_tokens = tf.tile(tf.constant([start_of_sequence_id], tf.inf32), [batch_size], name='start_tokens')

	inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
															   start_tokens,
															   end_of_sequence_id)

	inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
														inference_helper,
														encoder_state,
														output_layer)

	inference_output = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
														 impute_finished=True,
														 maximum_iterations=max_target_sequence_length)[0]

	return inference_output

# DECODING LAYER

def decoding_layer(dec_input,
				   encoder_state,
				   target_sequence_length,
				   max_target_sequence_length,
				   rnn_size,
				   num_layers,
				   target_vocab_to_int,
				   target_vocab_size,
				   batch_size,
				   keep_prob,
				   decoding_embedding_size):

	dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))

	dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

	dec_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(rnn_size),output_keep_prob=keep_prob) for _ in range(num_layers)])

	output_layer = tf.layers.Dense(target_vocab_size)

	with tf.variable_scope('decode'):
		decoder_training_output = decoding_layer_train(encoder_state,
													   dec_cell,
													   dec_embedd_input,
													   target_sequence_length,
													   max_target_sequence_length,
													   output_layer,
													   keep_prob)

	with tf.variable_scope('decode', reuse=True):
		decoder_inference_output = decoding_layer_infer(encoder_state,
														dec_cell,
														dec_embeddings,
														target_vocab_to_int['<GO>'],
														target_vocab_to_int['<EOS>'],
														max_target_sequence_length,
														target_vocab_size,
														output_layer,
														batch_size,
														keep_prob)


# BUILD THE NEURAL NETWORK

def seq2seq_model(input_data,
				  target_data,
				  keep_prob,
				  batch_size,
				  source_sequence_length,
				  target_sequence_length,
				  max_target_sequence_length,
				  source_vocab_size,
				  target_vocab_size,
				  enc_embedding_size,
				  dec_embedding_size,
				  rnn_Size,
				  num_layers,
				  target_vocab_to_int):

	encoder_outputs, encoder_state = encoding_layer(input_data,
													rnn_size,
													num_layers,
													keep_prob,
													source_sequence_length,
													source_vocab_size,
													enc_embedding_size)

	dec_input = process_decoder(target_data, target_vocab_to_int, bacth_size)

	training_decoder_outputs, inference_decoder_output = decoding_layer(dec_input,
																		encoder_state,
																		target_sequence_length,
																		max_target_sequence_length,
																		rnn_size,
																		num_layers,
																		target_vocab_to_int,
																		target_vocab_size,
																		batch_size,
																		keep_prob,
																		dec_embedding_size)

	return training_decoder_outputs, inference_decoder_output



# Number of Epochs
epochs = 5
# Batch Size
batch_size = 256
# RNN Size
rnn_size = 512
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 200
decoding_embedding_size = 200
# Learning Rate
learning_rate = 0.001
# Dropout Keep Probability
keep_probability = 0.75
display_step = 50




