from gensim.models.word2vec import Word2Vec
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D
from types import ListType

import hyper_params

class SingleLayerCNN(object):
    """docstring for SingleLayerCNN."""
    def __init__(self, seq_len, emb_dim, filter_len, feature_maps, w2v_model_path):
        super(SingleLayerCNN, self).__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.filter_len = filter_len
        self.feature_maps = feature_maps
        self.w2v = None
        self.model = self.__build_cnn__(w2v_model_path)

    def __build_conv_layer__(self):
        layer_name = "embedding_input"
        emb_input = Input(shape=(self.seq_len, self.emb_dim), dtype='int32', name=layer_name)
        conv_layers = []
        assert type(self.filter_len) is ListType, \
            "filter_len=%s is not a list" % (str(self.filter_len))

        for filter_length in self.filter_len:
            conv = Convolution1D(
                nb_filter=self.feature_maps,
                filter_length=filter_length,
                activation='tanh')(emb_input)
            conv = MaxPooling1D(self.seq_len - filter_length)(conv)
            conv = Flatten()(conv)
            conv_layers.append(conv)

        conv = Merge(mode='concat')(conv_layers) if len(conv_layers) > 1 else conv_layers[0]
        conv = Model(input=emb_input, output=conv, name='conv_layer')
        return conv

    def __build_cnn__(self, w2v_model_path):
        conv = self.__build_conv_layer__()
        main_input = Input(shape=(self.seq_len,), dtype='int32', name='main_input')
        self.w2v = Word2Vec.load(w2v_model_path)
        vocab_size = len(self.w2v.vocab)

        embedding = Embedding(
            input_dim=vocab_size,
            output_dim=self.emb_dim,
            input_length=self.seq_len,
            weights=[self.w2v.syn0])(main_input)

        conv = conv(embedding)
        conv = Dropout(.5)(conv)
        conv = Dense(2, activation="softmax")(conv)
        model = Model(input=main_input, output=conv)
        model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

        return model

    def word2index(self):
        return {word: (i + 1) for i, word in enumerate(self.w2v.index2word)}
