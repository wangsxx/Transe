"""
@File  : TransE.py
@Time  : 2019/10/23 11:05
@Author: Wangs
@Decs  : TransE模型的主要内容，使用TensorFlow实现
"""
import timeit
import tensorflow as tf
import math
from data_precess import KnowledgeGraph
import numpy as np


class TransE:
    def __init__(self, kg: KnowledgeGraph, embedding_dim, margin_value, dissimilarity_func, batch_size, learning_rate):
        # 根据文献 lr = {0.001, 0.01, 0.1} embedding_dim = {20, 50} margin_value = 1
        # WN18数据集 lr = 0.01  dim = 50  d_f = L1 margin_value = 1
        # FB15K lr = 0.01  dim = 50  d_f = L2 margin_value = 1
        self.kg = kg  # 知识图谱三元组
        self.embedding_dim = embedding_dim  # 嵌入维度
        self.margin_value = margin_value  # 取1
        self.dissimilarity_func = dissimilarity_func  # 不相似函数(稀疏函数) 一般取 L1 或 L2
        self.batch_size = batch_size  # batch_size
        self.learning_rate = learning_rate  # 学习率
        # 初始化正三元组和负三元组 (head_id, relation_id, tail_id)
        self.pos_triple = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.neg_triple = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.train_op = None  # 训练操作，是最小化优化器的那一步 如 tf.train.AdamOptimizer().minimize()
        self.loss = None  # 损失函数
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')  # 全局训练步数
        # 初始embedding
        self.entity_embedding = None
        self.relation_embedding = None
        self.init_embedding()  # 初始词嵌入
        self.init_train()  # 初始化训练步骤
        # 评估操作
        self.eval_triple = tf.placeholder(dtype=tf.int32, shape=[3])  # 评估三元组，3行n列
        self.idx_head_prediction = None
        self.idx_tail_prediction = None
        self.build_eval_graph()

    def init_embedding(self):
        # 初始化embedding，相当于初始化权重
        bound = 6 / math.sqrt(self.embedding_dim)
        self.entity_embedding = tf.get_variable(name="entity_embedding",
                                                shape=[self.kg.n_entity, self.embedding_dim],
                                                initializer=tf.random_uniform_initializer(-bound, bound))

        self.relation_embedding = tf.get_variable(initializer=tf.random_uniform_initializer(-bound, bound),
                                                  shape=[self.kg.n_relation, self.embedding_dim],
                                                  name="relation_embedding")

    def init_train(self):
        # 正向传播 计算损失 训练步骤
        # 正则化
        self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, dim=1)
        self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, dim=1)

        # 计算正/负样本的距离
        distance_pos, distance_neg = self.cal_distance(self.pos_triple, self.neg_triple)

        # 计算损失 与选择的dissimilarity_func有关
        if self.dissimilarity_func == 'L1':
            # 将每一行相加 所有pos/neg的得分
            score_pos = tf.reduce_sum(tf.abs(distance_pos), axis=1)
            score_neg = tf.reduce_sum(tf.abs(distance_neg), axis=1)
        else:  # L2 score
            score_pos = tf.reduce_sum(tf.square(distance_pos), axis=1)
            score_neg = tf.reduce_sum(tf.square(distance_neg), axis=1)

        # 使用relu函数取正数部分
        self.loss = tf.reduce_sum(tf.nn.relu(self.margin_value + score_pos - score_neg))
        # 选择优化器，并继续训练操作
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).\
            minimize(self.loss, global_step=self.global_step)

    def cal_distance(self, pos_triple, neg_triple):
        # tf.nn.embedding_lookup 从所有的embedding中选取 某几行的embedding  pos_triple中存的是下标
        pos_head = tf.nn.embedding_lookup(self.entity_embedding, pos_triple[:, 0])
        pos_rel = tf.nn.embedding_lookup(self.relation_embedding, pos_triple[:, 1])
        pos_tail = tf.nn.embedding_lookup(self.entity_embedding, pos_triple[:, 2])

        neg_head = tf.nn.embedding_lookup(self.entity_embedding, neg_triple[:, 0])
        neg_rel = tf.nn.embedding_lookup(self.relation_embedding, neg_triple[:, 1])
        neg_tail = tf.nn.embedding_lookup(self.entity_embedding, neg_triple[:, 2])

        # 距离维度 [1, embedding_dim]
        distance_pos = pos_head + pos_rel - pos_tail
        distance_neg = neg_head + neg_rel - neg_tail
        return distance_pos, distance_neg

    def launch_training(self, sess):
        # 开始训练
        start = timeit.default_timer()
        n_batch = 0
        # 构造batch
        epoch_loss = 0
        n_used_triple = 0
        for batches in self.kg.next_batch(self.batch_size):
            n_batch += 1
            batch_pos, batch_neg = self.kg.get_pos_neg(batches)
            batch_loss, _ = sess.run(fetches=[self.loss, self.train_op], feed_dict={self.pos_triple: batch_pos,
                                                                                    self.neg_triple: batch_neg})
            epoch_loss += batch_loss
            n_used_triple += len(batch_pos)
            print('[{:.3f}s] #triple: {}/{} triple_avg_loss: {:.6f}'.format(timeit.default_timer() - start,
                                                                            n_used_triple,
                                                                            self.kg.n_train_triple,
                                                                            batch_loss / len(batch_pos)))
        print('epoch loss: {:.3f}'.format(epoch_loss))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print('-----Finish training-----')

    def build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.idx_head_prediction, self.idx_tail_prediction = self.evaluate(self.eval_triple)

    def evaluate(self, eval_triple):
        with tf.name_scope('lookup'):
            head = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[0])  # 头结点
            tail = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[2])  # 尾结点
            relation = tf.nn.embedding_lookup(self.relation_embedding, eval_triple[1])  # 关系

        with tf.name_scope('link'):
            # 头结点预测 通过训练的节点嵌入来代替原始头结点
            distance_head_prediction = self.entity_embedding + relation - tail

            # 尾结点预测 通过训练的节点嵌入来代替原始尾结点
            distance_tail_prediction = head + relation - self.entity_embedding

        with tf.name_scope('rank'):
            if self.dissimilarity_func == 'L1':  # L1 score
                # tf.nn.top_k 返回 input 中每行最大的 k 个数，并且返回它们所在位置的索引
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
            else:  # L2 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
        return idx_head_prediction, idx_tail_prediction

    def launch_evaluation(self, session):
        print('-----Start evaluation-----')
        start = timeit.default_timer()
        rank_result_queue = []
        n_used_eval_triple = 0
        for eval_triple in self.kg.test_triple:
            idx_head_prediction, idx_tail_prediction = session.run(fetches=[self.idx_head_prediction,
                                                                            self.idx_tail_prediction],
                                                                   feed_dict={self.eval_triple: eval_triple})
            eval_result_queue = [eval_triple, idx_head_prediction, idx_tail_prediction]
            rank_result_queue.append(eval_result_queue)
            n_used_eval_triple += 1
            print('[{:.3f}s] #evaluation triple: {}/{}'.format(timeit.default_timer() - start,
                                                               n_used_eval_triple,
                                                               self.kg.n_test_triple))


        '''Raw'''
        head_meanrank_raw = 0
        head_hits10_raw = 0
        tail_meanrank_raw = 0
        tail_hits10_raw = 0
        '''Filter'''
        head_meanrank_filter = 0
        head_hits10_filter = 0
        tail_meanrank_filter = 0
        tail_hits10_filter = 0
        print(n_used_eval_triple, rank_result_queue)
        for i in range(n_used_eval_triple):
            head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter = self.calculate_rank(rank_result_queue[i])
            head_meanrank_raw += head_rank_raw
            if head_rank_raw < 10:
                head_hits10_raw += 1
            tail_meanrank_raw += tail_rank_raw
            if tail_rank_raw < 10:
                tail_hits10_raw += 1
            head_meanrank_filter += head_rank_filter
            if head_rank_filter < 10:
                head_hits10_filter += 1
            tail_meanrank_filter += tail_rank_filter
            if tail_rank_filter < 10:
                tail_hits10_filter += 1
        print('-----Raw-----')
        head_meanrank_raw /= n_used_eval_triple
        head_hits10_raw /= n_used_eval_triple
        tail_meanrank_raw /= n_used_eval_triple
        tail_hits10_raw /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_raw, head_hits10_raw))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_raw, tail_hits10_raw))
        print('------Average------')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_raw + tail_meanrank_raw) / 2,
                                                         (head_hits10_raw + tail_hits10_raw) / 2))
        print('-----Filter-----')
        head_meanrank_filter /= n_used_eval_triple
        head_hits10_filter /= n_used_eval_triple
        tail_meanrank_filter /= n_used_eval_triple
        tail_hits10_filter /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_filter, head_hits10_filter))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_filter, tail_hits10_filter))
        print('-----Average-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_filter + tail_meanrank_filter) / 2,
                                                         (head_hits10_filter + tail_hits10_filter) / 2))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print('-----Finish evaluation-----')

    def calculate_rank(self, idx_predictions):
        eval_triple, idx_head_prediction, idx_tail_prediction = idx_predictions
        head, relation, tail= eval_triple
        head_rank_raw = 0
        tail_rank_raw = 0
        head_rank_filter = 0
        tail_rank_filter = 0
        # idx_head_prediction[::-1] 倒序复制一遍 之前是从大到小 倒序后从小到大
        for candidate in idx_head_prediction[::-1]:
            if candidate == head:
                break
            else:
                head_rank_raw += 1
                if (candidate, relation, tail) in self.kg.golden_triple_pool:
                    continue
                else:
                    head_rank_filter += 1
        for candidate in idx_tail_prediction[::-1]:
            if candidate == tail:
                break
            else:
                tail_rank_raw += 1
                if (head, relation, candidate) in self.kg.golden_triple_pool:
                    continue
                else:
                    tail_rank_filter += 1
        return (head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter)


kg = KnowledgeGraph('data/WN18/')
kge_model = TransE(kg=kg, embedding_dim=200, margin_value=1.0, dissimilarity_func='L1', batch_size=4800, learning_rate=0.001)

gpu_config = tf.GPUOptions(allow_growth=True)
sess_config = tf.ConfigProto(gpu_options=gpu_config)

with tf.Session(config=sess_config) as sess:
    tf.global_variables_initializer().run()
    for epoch in range(100):
        print('=' * 30 + '[EPOCH {}]'.format(epoch) + '=' * 30)
        kge_model.launch_training(sess)
        if (epoch + 1) % 100 == 0:
            kge_model.launch_evaluation(sess)

