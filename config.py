# —*- coding: utf-8 -*-


class Config(object):
    def __init__(self, word_embedding_dimension=100, word_num=20000,
                 epoch=2000, sentence_max_size=40,
                 learning_rate=0.01, batch_size=1,
                 drop_out=0.5,
                 dict_size=50000,
                 bidirectional=False,
                 doc_len=40):
        self.word_embedding_dimension = word_embedding_dimension
        self.word_num = word_num
        self.epoch = epoch
        self.sentence_max_size = sentence_max_size                   # 句子长度
        self.lr = learning_rate
        self.batch_size = batch_size
        self.dict_size = dict_size
        self.drop_out = drop_out
        self.bidirectional = bidirectional
        self.doc_len = doc_len

