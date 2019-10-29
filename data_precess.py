"""
@File  : data_precess.py
@Time  : 2019/10/23 9:26
@Author: Wangs
@Decs  : 预处理数据文件，构建知识图谱
"""
import os
import random
import timeit
import pandas
import numpy as np

class KnowledgeGraph:
    def __init__(self, data_path):
        self.data_path = data_path  # 数据集文件夹的路径
        # 实体
        self.entity_dict = {}  # 存储所有实体信息 {'实体1': 下标1}
        self.entity = []  # 存储所有实体的下标， 不用self.entity_dict.value() 是因为这个操作特别耗时，4800-1.5s左右
        self.n_entity = 0  # 实体的数量
        # 关系
        self.relation_dict = {}  # 存储所有关系信息 {'关系1': 下标1}
        self.n_relation = 0  # 关系数量
        # 三元组 (head, relation, tail)
        self.train_triple = []  # 训练集三元组
        self.valid_triple = []  # 验证集三元组
        self.test_triple = []  # 测试集三元组
        self.n_train_triple = 0  # 训练集数量
        self.n_valid_triple = 0  # 验证集数量
        self.n_test_triple = 0  # 测试集数量
        # 加载dict和triple
        self.load_dict()
        self.load_triple()
        # 三元组池
        self.training_triple_pool = set(self.train_triple)
        self.golden_triple_pool = set(self.train_triple) | set(self.valid_triple) | set(self.test_triple)

    def load_dict(self):
        # 加载实体和关系字典
        entity_path = '/entity2id.txt'
        relation_path = '/relation2id.txt'
        # 第一列为名称 第二列为id
        print("="*20 + "load entity and relation dict" + "="*20)
        entity_file = pandas.read_table(self.data_path+entity_path, header=None)
        self.entity_dict = dict(zip(entity_file[0], entity_file[1]))
        self.entity = list(self.entity_dict.values())
        self.n_entity = len(self.entity_dict)
        print("# the number of entity: {}".format(self.n_entity))

        # 关系
        relation_file = pandas.read_table(self.data_path + relation_path, header=None)
        self.relation_dict = dict(zip(relation_file[0], relation_file[1]))
        self.n_relation = len(self.relation_dict)
        print("# the number of relation: {}".format(self.n_relation))

    def load_triple(self):
        # 加载三元组(head, relation, tail)
        train_path = "/train.txt"
        valid_path = "/valid.txt"
        test_path = "/test.txt"
        # 加载三元组，文件格式 (head_name, tail_name, relation_name)
        print("=" * 20 + "load train/valid/test triple" + "=" * 20)

        # 训练集
        train_file = pandas.read_table(self.data_path + train_path, header=None)
        # 三元组存储下标(id) (head_id, relation_id, tail_id)
        self.train_triple = list(zip([self.entity_dict[h] for h in train_file[0]],
                                     [self.relation_dict[r] for r in train_file[2]],
                                     [self.entity_dict[t] for t in train_file[1]]))
        self.n_train_triple = len(self.train_triple)
        print("# the number of train_triple: {}".format(self.n_train_triple))

        # 验证集
        valid_file = pandas.read_table(self.data_path + valid_path, header=None)
        # 三元组存储下标(id) (head_id, relation_id, tail_id)
        self.valid_triple = list(zip([self.entity_dict[h] for h in valid_file[0]],
                                     [self.relation_dict[r] for r in valid_file[2]],
                                     [self.entity_dict[t] for t in valid_file[1]]))
        self.n_valid_triple = len(self.valid_triple)
        print("# the number of valid triple: {}".format(self.n_valid_triple))

        # 测试集
        test_file = pandas.read_table(self.data_path + test_path, header=None)
        # 三元组存储下标(id) (head_id, relation_id, tail_id)
        self.test_triple = list(zip([self.entity_dict[h] for h in test_file[0]],
                                     [self.relation_dict[r] for r in test_file[2]],
                                     [self.entity_dict[t] for t in test_file[1]]))
        self.n_test_triple = len(self.test_triple)
        print("# the number of test triple: {}".format(self.n_test_triple))

    def next_batch(self, batch_size):
        rand_idx = np.random.permutation(self.n_train_triple)
        start = 0
        while start < self.n_train_triple:
            end = min(start + batch_size, self.n_train_triple)
            # yield 关键字 返回一个迭代对象
            yield [self.train_triple[i] for i in rand_idx[start:end]]
            start = end

    def get_pos_neg(self, batches_triple):

        batch_pos = batches_triple
        batch_neg = []
        # 随机试验 有0.5的概率为0 0.5的概率为1
        corrupt_head_prob = np.random.binomial(1, 0.5)
        for head, relation, tail in batch_pos:
            head_neg = head
            tail_neg = tail
            while True:
                if corrupt_head_prob:
                    # 如果为1 则是打乱头结点
                    # random.choice(self.entities) 从实体中随机选择一个
                    # self.entity_dict.values()是一个耗时的操作 因此提前准备好entity的数组
                    head_neg = random.choice(self.entity)

                else:
                    # 如果为0 则打乱尾结点
                    tail_neg = random.choice(self.entity)
                if (head_neg, relation, tail_neg) not in self.training_triple_pool:
                    break
            batch_neg.append((head_neg, relation, tail_neg))

        return batch_pos, batch_neg

# kg = KnowledgeGraph(data_path='data/WN18')
#
# for batches in kg.next_batch(4800):
#     start = timeit.default_timer()
#     batch_pos, batch_neg = kg.get_pos_neg(batches)
#     print(timeit.default_timer() - start)
