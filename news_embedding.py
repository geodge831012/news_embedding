# -*- coding: utf-8 -*-

import os
import numpy as np
import jieba_fast
from gensim.models import Word2Vec
import pickle



#########################################################################
## 求X Y两个向量的cosine值
def get_cosine_value(X, Y):
    # 分子 x1*y1 + x2*y2 + ... + xn*yn
    # 分母 ||X|| * ||Y||

    #print(X)
    #print(Y)

    if (np.linalg.norm(X) <= 0.0 or np.linalg.norm(Y) <= 0.0):
        return 0

    X = X.reshape(1, 256)
    Y = Y.reshape(1, 256)

    return float(X.dot(Y.T) / (np.linalg.norm(X) * np.linalg.norm(Y)))


##################################一个聚类簇的信息#######################################
class Clustering():
    def __init__(self):

        # 聚类簇的质心
        self.clustering_centroid = np.empty([0, 256], dtype=np.float)

        # 聚类簇的所有成员(以资讯全属性dict的形式存储,包括id、title、news_embedding等)
        self.news_info_list = []

        # 判断距离阈值 想要加入此聚类 需要和质心的距离大于distance_threshold才行
        self.distance_threshold = 0.9


    # 返回此簇的资讯数量
    def get_cluserting_news_size(self):

        return len(self.news_info_list)


    # 重新计算质心 同时保存资讯信息
    def reset_clustering_centroid(self, news_info):

        # 质心清空 初始化
        self.clustering_centroid = news_info["news_embedding"]

        # 重新计算质心
        for i in range(len(self.news_info_list)):
            self.clustering_centroid += self.news_info_list[i]["news_embedding"]

        # 保存资讯信息
        self.news_info_list.append(news_info)


    # 判断一个资讯能不能加入一个簇
    def calc_news_cluserting(self, news_info):

        # 簇为空 或者 簇心和新资讯的距离足够近  都可以加入到簇中
        if 0 == self.get_cluserting_news_size() or \
                get_cosine_value(self.clustering_centroid, news_info["news_embedding"]) >= self.distance_threshold:
            self.reset_clustering_centroid(news_info)
            return 0

        # 不成功 返回1
        return 1



##################################处理资讯嵌入的类#######################################
class NewsEmbedding():
    def __init__(self):

        print("NewsEmbedding __init__")

        # word2vec模型
        self.model = Word2Vec.load("word2vec/word2vec_wx")

        # 所有句子的向量集合 Vs
        self.all_news_dict = {}



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

        key_list = news_info_dict["key_list"]
        for i in range(len(key_list)):

            # 单词名
            word        = key_list[i][0]
            # 此词的tfidf的值
            word_tfidf  = float(key_list[i][1])

            # word2vec某词
            w2v_vector = self.get_word2vec(word)
            if 0 == len(w2v_vector):
                continue

            coe = word_tfidf

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

            self.all_news_dict[news_id] = news_info_dict





#########################################################################
## main 主函数 ##

if __name__ == '__main__':

    news_embedding_class = NewsEmbedding()

    # 处理新闻数据
    news_embedding_class.proc_news_data()


    # 聚类簇的集合
    clustering_list = []

    # 遍历所有的资讯
    for news_id, news_info in news_embedding_class.all_news_dict.items():

        # 遍历所有的簇  看有无合适的簇可以加入此篇资讯
        is_append_clustering = False

        for i in range(len(clustering_list)):
            if 0 == clustering_list[i].calc_news_cluserting(news_info):
                # 成功加入簇
                is_append_clustering = True
                break

        if False == is_append_clustering:
            # 没有加入到任何一个簇 则新起一个簇
            new_clustering = Clustering()
            new_clustering.calc_news_cluserting(news_info)

            clustering_list.append(new_clustering)



    print("=================print_result=======================")
    print("clustering_list.len=%d" % len(clustering_list))
    print("=================print_result=======================")

    for i in range(len(clustering_list)):
        if len(clustering_list[i].news_info_list) > 3:
            news_info_list = clustering_list[i].news_info_list

            for j in range(len(news_info_list)):
                print(news_info_list[j]["title"])

            print("===================%d=====================" % i)



