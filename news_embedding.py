# -*- coding: utf-8 -*-

from gensim.models import Word2Vec

import os
import numpy as np
import pickle
import func_utils




##################################处理资讯嵌入的类#######################################
class NewsEmbedding():
    def __init__(self):

        print("NewsEmbedding __init__")

        # word2vec模型
        self.model = Word2Vec.load("word2vec/word2vec_wx")

        # 所有资讯的集合
        self.all_news_list = []



    # 获取某个词的word2vec值 256维
    def get_word2vec(self, word):

        if(word in self.model):
            return self.model[word]
        else:
            return np.zeros(0, dtype=np.float)


    # 通过一篇资讯新闻的信息 获取到的news_embedding
    def get_news_embedding(self, news_info_dict):

        # 句子的向量 Vs
        news_embedding = np.zeros(self.model.vector_size, dtype=np.float)

        key_list = eval(news_info_dict["key_words"])
        for i in range(len(key_list)):

            # 单词名
            word        = key_list[i][0]
            # 此词的tfidf的值
            word_tfidf  = key_list[i][1]

            # word2vec某词
            w2v_vector = self.get_word2vec(word)
            if 0 == len(w2v_vector):
                continue

            coe = func_utils.sigmoid(word_tfidf)

            news_embedding = news_embedding + coe * w2v_vector

        return news_embedding


    # 处理新闻数据
    def proc_news_data(self):

        print("=====================news_embedding_class.proc_news_data======================")

        all_news_dict = pickle.load(open("./all_news.dat", "rb"))

        # 遍历所有资讯
        for news_id, news_info_dict in all_news_dict.items():
            news_embedding = self.get_news_embedding(news_info_dict)

            news_info_dict["news_embedding"] = news_embedding

            # 计算资讯向量的norm值(开方(x1方 + x2方 + ... + xn方))，用于后面的夹角余弦的计算
            news_info_dict["news_norm"] = np.linalg.norm(news_embedding)

            self.all_news_list.append(news_info_dict)





#########################################################################
## main 主函数 ##

if __name__ == '__main__':

    news_embedding_class = NewsEmbedding()

    # 处理新闻数据
    news_embedding_class.proc_news_data()



