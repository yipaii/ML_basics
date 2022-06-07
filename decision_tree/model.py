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


class DT:
    def __init__(self, dim, k):
        self.dim = dim
        self.num_classes = k
        pass

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
            for option in available_feature[i]:
                gd[i] -= (sample_cnt_of_cur_feature[option] / total_num) * cal_hd(data_partition[option])
        return gd