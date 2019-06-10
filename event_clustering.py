# -*- coding: utf-8 -*-

from tqdm import tqdm
from acora import AcoraBuilder
from news_embedding import NewsEmbedding

import os
import numpy as np
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

        # 股票名称过滤的阈值
        self.filter_threshold = 0.8

        # 所有股票列表
        self.stock_list = []
        for line in open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/stocklist.txt")):
            self.stock_list.append(line.strip())


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


    # 主题进行过滤 主要是过滤一季度报等信息
    # 如果同一个簇的主题列表包含过多的股票名称 则过滤之
    def filter_clustering(self):

        # select SecuAbbr from secumain where SecuMarket in (83,90) and SecuCategory=1 and ListedSector in (1,2,6) and ListedState=1

        # build AC model
        builder = AcoraBuilder()

        for i in range(len(self.stock_list)):
            builder.add(self.stock_list[i])

        tree = builder.build()

        #返回的聚类list
        rst_clustering_list = []

        # 处理聚类
        for i in range(len(self.clustering_list)):
            # 一个聚类
            clustering = self.clustering_list[i]

            # 此聚类主题下的资讯篇数
            news_num = len(clustering.news_info_list)

            content_str = ""
            for j in range(len(clustering.news_info_list)):
                news_info = clustering.news_info_list[j]
                content_str += news_info["news_title"]

            # 匹配到的关键词 要进行唯一化处理
            # 比如10篇文章都有华谊兄弟亏损的 就不应该过滤
            # 如果是多个公司一季度报的 就需要过滤
            unique_word_set = set()

            for hit_word, pos in tree.finditer(content_str):
                unique_word_set.add(hit_word)

            if len(unique_word_set) / news_num < self.filter_threshold:
                rst_clustering_list.append(clustering)

        self.clustering_list = rst_clustering_list



    # 聚类信息打印
    def print_clustering(self):

        print("=================print_result=======================")
        print("clustering_mgr.clustering_list.len=%d" % len(clustering_mgr.clustering_list))
        print("=================print_result=======================")

        for i in range(len(clustering_mgr.clustering_list)):
            if len(clustering_mgr.clustering_list[i].news_info_list) > 3:
                news_info_list = clustering_mgr.clustering_list[i].news_info_list

                for j in range(len(news_info_list)):
                    print(news_info_list[j]["news_title"])

                print("===================%d=====================" % i)


#########################################################################
## main 主函数 ##

if __name__ == '__main__':

#    clustering_mgr = pickle.load(open("cluster.dat", "rb"))
#    clustering_mgr.print_clustering()
#    exit(0)

    news_embedding = NewsEmbedding()

    # 处理新闻数据
    news_embedding.proc_news_data()


    print("news_embedding_class.all_news_list.len=%d" % len(news_embedding.all_news_list))
    # 遍历所有的资讯 做聚类
    clustering_mgr = ClusteringMgr()
    clustering_mgr.clustering_all_news(news_embedding.all_news_list)
    clustering_mgr.filter_clustering()


    clustering_mgr.print_clustering()

    # 存储聚类信息
    pickle.dump(clustering_mgr, open("./cluster.dat", "wb"))



