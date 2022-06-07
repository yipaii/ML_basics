import model


def train_dt(file_path):
    data = []
    classes = set()
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            line = list(map(int, line.strip().replace('\n', '').split(' ')))
            if not line or len(line) != 5:
                continue
            x, y = line[:-1], line[-1]
            if y not in classes:
                classes.add(y)
            data.append((x, y))
    dim = len(data[0][0])
    dt = model.DT(dim, len(classes))
    gd = dt.cal_information_gain(data)
    print(dt.select_feature(gd))


if __name__ == '__main__':
    train_dt('data.txt')
