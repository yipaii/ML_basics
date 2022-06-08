from functools import reduce
from math import log2


def cal_hd(data):
    total_num = len(data)
    cnt_each_class = dict()
    for x, y in data:
        if y in cnt_each_class:
            cnt_each_class[y] += 1
        else:
            cnt_each_class[y] = 1
    hd = 0
    for i in cnt_each_class:
        hd -= (cnt_each_class[i] / total_num) * log2(cnt_each_class[i] / total_num)

    return hd


class TreeNode:
    def __init__(self, feature=None, fv=None, samples=None):
        self.feature = feature
        self.fv = fv
        self.samples = samples
        self.children = []


class DT:
    def __init__(self, dim, k):
        self.dim = dim
        self.num_classes = k

    def select_feature(self, gd) -> int:
        best_feature = 0
        base = gd[0]
        for i in range(self.dim):
            if gd[i] > base:
                best_feature = i
                base = gd[i]
        return best_feature

    def cal_information_gain(self, data):
        total_num = len(data)
        gd = [0] * self.dim
        available_feature = {}
        hd = cal_hd(data)
        features = list(map(lambda x: x[0], data))
        zip_feature = list(zip(*features))
        samples_partition_of_each_dim = {}
        for i in range(self.dim):
            available_feature[i] = list(set(zip_feature[i]))
            data_partition = {}
            sample_cnt_of_cur_feature = {}
            for x, y in data:
                if x[i] in sample_cnt_of_cur_feature:
                    sample_cnt_of_cur_feature[x[i]] += 1
                    data_partition[x[i]].append((x, y))
                else:
                    sample_cnt_of_cur_feature[x[i]] = 1
                    data_partition[x[i]] = [(x, y)]
            gd[i] = hd
            samples_partition_of_each_dim[i] = data_partition
            for option in available_feature[i]:
                gd[i] -= (sample_cnt_of_cur_feature[option] / total_num) * cal_hd(data_partition[option])
        return gd, available_feature, samples_partition_of_each_dim

    def ID3(self, feature=None, fv=None, samples=None) -> TreeNode:
        sample_num = len(samples)
        num_labels = 0
        labels = set()
        for _, y in samples:
            if y not in labels:
                labels.add(y)
                num_labels += 1
        if sample_num == 1 or num_labels == 1:
            return TreeNode(feature=feature, fv=fv, samples=samples)
        res = TreeNode(feature=feature, fv=fv, samples=samples)
        gd, available_feature, samples_partition_of_each_dim = self.cal_information_gain(data=samples)
        feature = self.select_feature(gd=gd)
        for i in available_feature[feature]:
            samples_partition = samples_partition_of_each_dim[feature][i]
            child = self.ID3(feature=feature, fv=i, samples=samples_partition)
            res.children.append(child)
        return res

    def cal_cost(self, root):
        pass

    def prune(self, root):
        pass