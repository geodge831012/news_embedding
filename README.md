# news_embedding
获取一篇资讯的embedding表达



主要思路:

1.通过TF-IDF算法获取一篇资讯的关键字排名，获取此排名前若干个关键词(3个？5个？)，同时获取对应关键词的TF-IDF值

2.通过word2vec获取每个词的词向量，每个词的TF-IDF值作为系数，累加所有关键词的词向量，得到本篇资讯的news embedding
具体公式如下：
news_embedding = TF-IDF(word_1)*W2V(word_1) + TF-IDF(word_2)*W2V(word_2) + ... + TF-IDF(word_n)*W2V(word_n)

3.通过聚类算法获取相似的资讯，聚集在一起，使用余弦相似性计算。当一篇资讯和某个聚类簇的相似性超过阈值的时候，则将这篇资讯加入到这个聚类簇中，同时更新这个簇的质心



注意事项：

1.工程中的all_news.dat，已经是按照主要思路中的第1步预处理的结果数据了，pickle格式存储
