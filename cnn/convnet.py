from gensim.models.word2vec import Word2Vec
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D

import hyper_params

conv_layers = []

emb_input = Input(
    shape=(hyper_params.SEQUENCE_LENGTH, hyper_params.EMBEDDING_DIM),
    dtype='int32',
    name='emb_input')

for filter_length in hyper_params.FILTER_LENGTHS:
    conv = Convolution1D(
        nb_filter=hyper_params.FEATURE_MAPS,
        filter_length=filter_length,
        activation='tanh')(emb_input)
    conv = MaxPooling1D(hyper_params.SEQUENCE_LENGTH - filter_length)(conv)
    conv = Flatten()(conv)
    conv_layers.append(conv)

conv = Merge(mode='concat')(conv_layers)
conv = Model(input=emb_input, output=conv, name='conv_layer')

# embedding
w2v = Word2Vec.load('../models/76d3fa_2016117T2323.model')
vocab_size = len(w2v.vocab)

main_input = Input(shape=(hyper_params.SEQUENCE_LENGTH,), dtype='int32', name='main_input')
embedding = Embedding(
    input_dim=vocab_size,
    output_dim=hyper_params.EMBEDDING_DIM,
    input_length=hyper_params.SEQUENCE_LENGTH,
    weights=[w2v.syn0])(main_input)
conv = conv(embedding)
conv = Dropout(.5)(conv)
conv = Dense(2, activation="softmax")(conv)
model = Model(input=main_input, output=conv)
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.summary()
