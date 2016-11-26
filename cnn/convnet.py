from gensim.models.word2vec import Word2Vec
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D
from types import ListType

import hyper_params
import ujson as json
from time import asctime

class SingleLayerCNN(object):
    """docstring for SingleLayerCNN."""
    def __init__(self, seq_len, emb_dim, filter_len, feature_maps, w2v_model_path):
        super(SingleLayerCNN, self).__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.filter_len = filter_len
        self.feature_maps = feature_maps
        self.w2v = None
        self.w2v_model_path = w2v_model_path
        self.model = self.__build_cnn__(w2v_model_path)

    def describe_params(self):
        return {
            "seq_len": self.seq_len,
            "emb_dim": self.emb_dim,
            "filter_len": self.filter_len,
            "feature_maps": self.feature_maps,
            "w2v": self.w2v_model_path,
        }

    def save_model(self):
        now = asctime().replace(' ', '_')
        prefix = "cnn_{0}".format(now)
        self.model.save('../models/cnn/{0}.hd5'.format(prefix))
        with open('../models/cnn/{0}.json'.format(prefix), 'w') as f_config:
            metadata = self.describe_params()
            f_config.write(json.dumps(metadata))


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
                activation='relu')(emb_input)
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
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        return model

    def word2index(self):
        return {word: i for i, word in enumerate(self.w2v.index2word)}
