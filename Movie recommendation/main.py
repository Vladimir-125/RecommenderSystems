# -*- coding: utf-8 -*-
import topNrecommender as rec

def read_user_id():
    with open('input.txt', 'r') as f:
        return [l.strip() for l in  f.readlines()]


def write_output(prediction):
    with open('output.txt', 'w') as f:
        for p in prediction:
                f.write(p + "\n")

def do(ids):
    # test implementation
    #prediction = [['{},{},{}'.format(i, 5, 3.5)]*30 for i in ids]
    prediction = []
    for i in ids:
        recommends = rec.get_topN(int(i), 30)
        predics = [['{},{},{:.4f}'.format(int(e[0,0]), int(e[0,1]), e[0,2])] for e in recommends]
        for p in predics:
            prediction.append(str(p[0]))
    return prediction


if __name__ == "__main__":
    user_ids = read_user_id()
    result = do(user_ids)
    write_output(result)