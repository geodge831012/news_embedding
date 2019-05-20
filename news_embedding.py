# -*- coding: utf-8 -*-

import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm
import pickle
import func_utils



##################################一个聚类簇的信息#######################################
class Clustering():
    def __init__(self):

        # 聚类簇的质心
        self.clustering_centroid = np.empty([0, 256], dtype=np.float)

        # 聚类簇的所有成员(以资讯全属性dict的形式存储,包括id、title、news_embedding等)
        self.news_info_list = []

        # 簇心的norm值 方便做夹角余弦计算
        self.clustering_norm = 0.0


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

        # 重新计算质心的norm值
        self.clustering_norm = np.linalg.norm(self.clustering_centroid)

        # 保存资讯信息
        self.news_info_list.append(news_info)


class ClusteringMgr():
    def __init__(self):

        # 用于存储所有簇的质心
        self.g_clustering_centroid_matrix = np.empty([0, 256], dtype=np.float)

        # 判断距离阈值 想要加入此聚类 需要和质心的距离大于distance_threshold才行
        self.distance_threshold = 0.9

        # 聚类簇的集合
        self.clustering_list = []

    # 聚类所有的资讯
    def clustering_all_news(self, all_news_list):

        for news_info in tqdm(all_news_list):

            # 是否合并一个簇
            is_merge_clustering = False

            if (len(self.g_clustering_centroid_matrix) > 0):
                # x1*y1 + x2*y2 + ... + xn*yn / sqrt(x1*x1 + x2*x2 + xn*xn) * sqrt(y1*y1 + y2*y2 + yn*yn)
                result_list = list(func_utils.matrix_dot(self.g_clustering_centroid_matrix, news_info["news_embedding"]))

                for i in range(len(result_list)):
                    fenmu_float = self.clustering_list[i].clustering_norm * news_info["news_norm"]
                    if (fenmu_float > 0):
                        result_list[i] /= fenmu_float

                # 有超过阈值的结果
                if (max(result_list) > self.distance_threshold):
                    pos = result_list.index(max(result_list))

                    # 合并簇
                    self.clustering_list[pos].reset_clustering_centroid(news_info)
                    self.g_clustering_centroid_matrix[pos] = self.clustering_list[pos].clustering_centroid

                    is_merge_clustering = True


            if False == is_merge_clustering:
                # 如果不合并簇，那么就新增一个簇
                new_clustering = Clustering()
                new_clustering.reset_clustering_centroid(news_info)

                self.g_clustering_centroid_matrix = np.vstack((self.g_clustering_centroid_matrix, news_info["news_embedding"]))
                self.clustering_list.append(new_clustering)




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


    print("news_embedding_class.all_news_list.len=%d" % len(news_embedding_class.all_news_list))
    # 遍历所有的资讯 做聚类
    clustering_mgr = ClusteringMgr()
    clustering_mgr.clustering_all_news(news_embedding_class.all_news_list)


    print("=================print_result=======================")
    print("clustering_mgr.clustering_list.len=%d" % len(clustering_mgr.clustering_list))
    print("=================print_result=======================")

    for i in range(len(clustering_mgr.clustering_list)):
        if len(clustering_mgr.clustering_list[i].news_info_list) > 3:
            news_info_list = clustering_mgr.clustering_list[i].news_info_list

            for j in range(len(news_info_list)):
                print(news_info_list[j]["title"])

            print("===================%d=====================" % i)



