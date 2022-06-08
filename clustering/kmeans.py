from cmath import sqrt
import random
from math import inf


def cal_dist(x, y):
    n = len(x)
    dist = 0
    for i in range(n):
        dist += (x[i] - y[i]) ** 2
    return sqrt(dist).real


def update_center(centers_and_points):
    centers = []
    for i in centers_and_points:
        n = len(centers_and_points[i])
        centers.append(list(map(lambda x: sum(x) / n, list(zip(*centers_and_points[i])))))
    return centers


def cal_loss(dataset, centers):
    k = len(centers)
    loss = 0
    d = {}
    for i in range(k):
        d[i] = []
    for data in dataset:
        min_dist = inf
        belong = -1
        for i, center in enumerate(centers):
            dist = cal_dist(data, center)
            if dist < min_dist:
                min_dist = dist
                belong = i
        d[belong].append(data)
        loss += min_dist ** 2
    return d, loss


def kmeans(dataset, k):
    rand_centers = random.sample(dataset, k)
    old_loss = inf
    centers_and_points, loss = cal_loss(dataset, rand_centers)
    threshold = 1e-2
    times = 0
    epochs = 10
    while old_loss - loss > threshold and times < epochs:
        old_loss = loss
        new_centers = update_center(centers_and_points)
        centers_and_points, loss = cal_loss(dataset, new_centers)
        times += 1
    return centers_and_points, new_centers

if __name__ == '__main__':
    dataset = [[0, 0], [0, 1], [3, 3], [4, 7], [5, 8], [-2, -5]]
    k = 3
    print(kmeans(dataset, k))





