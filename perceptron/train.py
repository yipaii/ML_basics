import model


def train_model(file_path):
    data = []
    with open(file_path) as file:
        lines = file.readlines()
        for line in lines:
            if not line:
                continue
            line = line.strip().replace('\n', '').split(' ')
            x = list(map(int, line[:-1]))
            y = int(line[-1])
            data.append((x, y))
    dim = len(data[0][0])
    perceptron = model.Perceptron(dim, 2)
    epoch = 0
    while perceptron.cal_loss(data)[1] > 0:
        perceptron.train_param(data)
        epoch += 1
    print(f'weight: {perceptron.weight}\nbias: {perceptron.bias}\nepoch: {epoch}')


if __name__ == '__main__':
    train_model('data.txt')
