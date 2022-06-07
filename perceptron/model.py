class Perceptron:
    def __init__(self, dim, lr):
        self.lr = lr
        self.dim = dim
        self.weight = [1] * self.dim
        self.bias = 1

    def cal_loss(self, data):
        total_loss = 0
        mismatch_cnt = 0
        for x, y in data:
            res = sum(self.weight[i] * x[i] for i in range(self.dim)) + self.bias
            loss = res * y
            if loss <= 0:
                total_loss -= loss
                mismatch_cnt += 1
        return total_loss, mismatch_cnt

    def train_param(self, data):
        for x, y in data:
            res = sum(self.weight[i] * x[i] for i in range(self.dim)) + self.bias
            if res * y <= 0:
                for i in range(self.dim):
                    self.weight[i] += self.lr * y * x[i]
                self.bias += self.lr + y

    def predict(self, x):
        return 1 if sum(self.weight[i] * x[i] for i in range(self.dim)) + self.bias > 0 else -1
