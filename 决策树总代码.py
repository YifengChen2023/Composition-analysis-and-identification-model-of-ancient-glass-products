import csv
import copy

def loadDataset(filename):
    with open(filename, 'r') as f:
        lines = csv.reader(f)
        data_set = list(lines)

    # 整理数据
    category = data_set[0]
    del (data_set[0])

    for i in range(len(data_set)):
        for j in range(len(data_set[0])):
            data_set[i][j] = float(data_set[i][j])

    return data_set, category

def split_data(data): # 将数据与标签分离

    data_set = copy.deepcopy(data)

    data_mat = []
    label_mat = []
    for i in range(len(data_set)):
        label_mat.append(data_set[i][-1])
        del(data_set[i][-1])
        data_mat.append(data_set[i])

    return data_mat, label_mat


def get_best_split_value(data, result):
    # 找到当前特征下的最佳划分值
    data_set = copy.deepcopy(data)
    result_set = copy.deepcopy(result)
    lst = list(zip(data_set, result_set))
    lst.sort(key=lambda x: x[0])
    length = len(data_set)
    for i in range(length):
        data_set[i] = lst[i][0]
        result_set[i] = lst[i][1]

    if length == 1:
        return float("Inf"), float("Inf")

    split_value = []

    for i in range(length-1):
        split_value.append((data_set[i+1] + data_set[i]) / 2)

    gini = []

    for i in range(length - 1):
        gini.append(get_gini(i, result_set))

    min_gini = 0

    for i in range(len(gini)):
        if gini[i] < gini[min_gini]:
            min_gini = i

    return split_value[min_gini], gini[min_gini]

def get_gini(split_value_position, label_sorted):
    # 求出当前划分下的gini_index
    k_count = p_count = 0
    for i in range(split_value_position+1):
        if label_sorted[i] == 0:
            k_count += 1
        else:
            p_count += 1
    gini_1 = 1 - ((k_count / (k_count + p_count)) ** 2 + (p_count / (k_count + p_count)) ** 2)

    i = split_value_position + 1
    k_count = p_count = 0
    while i < len(label_sorted):
        if label_sorted[i] == 0:
            k_count += 1
        else:
            p_count += 1
        i += 1
    gini_2 = 1 - ((k_count / (k_count + p_count)) ** 2 + (p_count / (k_count + p_count)) ** 2)

    gini_gain = (split_value_position+1)/len(label_sorted) * gini_1 + (1 - (split_value_position+1)/len(label_sorted)) * gini_2

    return gini_gain

def get_best_feature(data, category):
    # 找到gini_index最小值的特征
    length = len(category)-1

    data_set, result = split_data(data)

    feature_gini = []

    split_feature_value = []

    feature_values = []

    for i in range(length):
        feature_gini.append(0)
        split_feature_value.append(0)

        feature_values.append([])
        for j in range(len(data_set)):
            feature_values[i].append(data_set[j][i])

    for i in range(length):
         split_feature_value[i], feature_gini[i] = get_best_split_value(feature_values[i], result)

    feature_num = 0
    # 找到最小gini_gain对应的feature编号
    for i in range(length):
        if feature_gini[i] < feature_gini[feature_num]:
            feature_num = i


    return feature_num, split_feature_value[feature_num]

class Node(object):
    def __init__(self, category, item):
        self.name = category
        self.elem = item
        self.lchild = None
        self.rchild = None

def leaf_value(data): # 返回叶子节点值
    sum = 0
    for i in range(len(data)):
        sum += data[i][-1]

    return sum/len(data)

def creat_tree(data, labels, feature_labels=[]):
    # 递归生成决策树
    # 结束条件
    if len(labels) == 1:
        return Node('result', leaf_value(data))
    if abs(leaf_value(data) - 1) < 1e-5:
        return Node('result', leaf_value(data))
    if abs(leaf_value(data)) < 1e-5:
        return Node('result', leaf_value(data))


    # 最优特征的标签
    best_feature_num, best_feature_value = get_best_feature(data, labels)

    feature_labels.append(labels[best_feature_num])

    node = Node(labels[best_feature_num], best_feature_value)

    ldata = []
    rdata = []
    i = 0
    for d in data:
        if d[best_feature_num] <= best_feature_value:
            ldata.append(d)
        else:
            rdata.append(d)
        del (d[best_feature_num])
        i+=1

    labels2 = copy.deepcopy(labels)
    del(labels2[best_feature_num])

    tree = node
    tree.lchild = creat_tree(ldata, labels2, feature_labels)
    tree.rchild = creat_tree(rdata, labels2, feature_labels)

    return tree

def breadth_travel(tree):
    # 广度遍历
    queue = [tree]
    while queue:
        cur_node = queue.pop(0)
        print(cur_node.name, end=" ")
        print(cur_node.elem, end=" ")
        if cur_node.lchild is not None:
            print('my lchild is', cur_node.lchild.name, cur_node.lchild.elem, end=' ')
            queue.append(cur_node.lchild)
        if cur_node.rchild is not None:
            print('my rchild is', cur_node.rchild.name, cur_node.rchild.elem, end=' ')
            queue.append(cur_node.rchild)
        print()


if __name__ == "__main__":
    # 装入数据
    train_set, category = loadDataset('decision_tree_-Pb_-Ba.csv')
    # 生成树
    my_tree = creat_tree(train_set, category)
    # 打印结果
    breadth_travel(my_tree)