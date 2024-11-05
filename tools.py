import torch
import dgl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC


def idx_to_one_hot(idx_arr):
    one_hot = np.zeros((idx_arr.shape[0], idx_arr.max() + 1))
    one_hot[np.arange(idx_arr.shape[0]), idx_arr] = 1
    return one_hot


def kmeans_test(X, y, n_clusters, repeat=10):#聚类数，也就是分成3类
    nmi_list = []
    ari_list = []
    for _ in range(repeat):#迭代进行训练次数
        kmeans = KMeans(n_clusters=n_clusters)#进行KMeans聚类
        y_pred = kmeans.fit_predict(X)#预测
        nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')#计算标准互信息值，NMI【0,1】
        ari_score = adjusted_rand_score(y, y_pred)#调整的兰德系数，ARI【-1,1】
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
    return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)#返回NMI和ARI的均值及标准差


def svm_test(X, y, test_sizes=(0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99), repeat=10):
    random_states = [182318 + i for i in range(repeat)]
    result_macro_f1_list = []
    result_micro_f1_list = []
    for test_size in test_sizes:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(repeat):#遍历重复次数
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_states[i])#按照test_sizes进行分割测试集
            svm = LinearSVC(dual=False)#线性分类支持向量机。dual为False解决原始问题，否则解决对偶问题
            svm.fit(X_train, y_train)#训练模型
            y_pred = svm.predict(X_test)#预测模型，返回预测值
            macro_f1 = f1_score(y_test, y_pred, average='macro')#测试集（这里指分割后的测试集）的损失与预测
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        result_macro_f1_list.append((np.mean(macro_f1_list), np.std(macro_f1_list)))#记录重复repeat次数下的均值和标准差
        result_micro_f1_list.append((np.mean(micro_f1_list), np.std(micro_f1_list)))
    return result_macro_f1_list, result_micro_f1_list#返回的是均值和标准差的列表


def evaluate_results_nc(embeddings, labels, num_classes):#输入的是测试集的嵌入，测试集的标签，类别的数量
    repeat = 10
    # print('SVM test')
    svm_macro_f1_list, svm_micro_f1_list = svm_test(embeddings, labels, repeat=repeat)#经过SVM进行测试20次，返回每次的评价列表
    #打印训练集比率为？？下的结果
    print('Macro-F1: ' + ', '.join(['{:.4f}~{:.4f}({:.2f})'.format(macro_f1_mean, macro_f1_std, train_size) for
                                    (macro_f1_mean, macro_f1_std), train_size in
                                    zip(svm_macro_f1_list, [0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01])]))
    print('Micro-F1: ' + ', '.join(['{:.4f}~{:.4f}({:.2f})'.format(micro_f1_mean, micro_f1_std, train_size) for
                                    (micro_f1_mean, micro_f1_std), train_size in
                                    zip(svm_micro_f1_list, [0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01])]))
    # print('\nK-means test')#得到的嵌入进行K-means测试
    nmi_mean, nmi_std, ari_mean, ari_std = kmeans_test(embeddings, labels, num_classes, repeat=repeat)
    # print('NMI: {:.6f}~{:.6f}'.format(nmi_mean, nmi_std))
    # print('ARI: {:.6f}~{:.6f}'.format(ari_mean, ari_std))

    macro_mean = [x for (x, y) in svm_macro_f1_list]
    micro_mean = [x for (x, y) in svm_micro_f1_list]
    return np.array(macro_mean), np.array(micro_mean), nmi_mean, ari_mean#返回各参数的均值


def parse_adjlist(adjlist, edge_metapath_indices, samples=None):

    edges = []
    nodes = set()
    result_indices = []
    for row, indices in zip(adjlist, edge_metapath_indices):#遍历图的邻接矩阵和图下的节点索引
        row_parsed = list(map(int, row.split(' ')))#将row中的数变成int并且组成list
        nodes.add(row_parsed[0])#nodes集合加入第一个元素，就是目标节点
        if len(row_parsed) > 1:#如果长度大于1，也就是说节点不是单独一个，存在与其他节点边相连
            if samples is None:#没有采样的话
                neighbors = row_parsed[1:]#节点的邻居为所有相连的节点
                result_indices.append(indices)#结果索引中添加所有的索引
            else:#如果有采样
                # 欠采样频繁邻居，主要思路是为了达到类别平衡的目的
                unique, counts = np.unique(row_parsed[1:], return_counts=True)#将节点的一阶邻居中重复的去掉，类似集合set,返回数在原数组中出现的次数为counts
                p = []
                #也就是说，unique是节点每个邻居（除了自己），counts是节点每个邻居的次数
                for count in counts:#遍历返回节点每个邻居的次数的值
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)#这里面的数应该是counts数值之和，也就是说，每个节点包括重复次数都出现一遍
                p = p / p.sum()#求和归一化，实际上出现的，也就是得到后续的采样概率
                samples = min(samples, len(row_parsed) - 1)#从节点采样数和节点邻居数中最小化一个出来
                sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))#从邻居中按照概率去选择样本，并进行排序返回索引
                neighbors = [row_parsed[i + 1] for i in sampled_idx]#按照采样后的规定新的节点邻居
                result_indices.append(indices[sampled_idx])#返回的是节点新的邻居下的索引
        else:#如果节点是孤立的节点
            neighbors = []
            result_indices.append(indices)
        for dst in neighbors:#遍历邻居
            nodes.add(dst)#节点增加他的邻居（79行代码中已经增加了节点它自己）
            edges.append((row_parsed[0], dst))#edges增加目标节点和新的邻居节点之间的边
    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}#将nodes排序后遍历返回下标(新排序后)和索引(实际节点索引)作为键值变成字典
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))#具体来说是将两节点之间变成边组成元组
    result_indices = np.vstack(result_indices)#按行堆叠
    return edges, result_indices, len(nodes), mapping#返回边的列表，按行进行排列A，节点的数量，下标和索引构成的字典


def parse_minibatch(adjlists, edge_metapath_indices_list, idx_batch, device, samples=None):
    g_list = []#图的列表
    result_indices_list = []#结果的索引列表
    idx_batch_mapped_list = []
    for adjlist, indices in zip(adjlists, edge_metapath_indices_list):#遍历两个元路径下的图和节点索引，长度4019
        edges, result_indices, num_nodes, mapping = parse_adjlist(
            [adjlist[i] for i in idx_batch], [indices[i] for i in idx_batch], samples)#这个调用的就是上面那个函数
        #上面输入的是子图，返回的是每个子图下对应的东西
        g = dgl.DGLGraph(multigraph=True)#创建一个多图
        g.add_nodes(num_nodes)#添加节点数（每张子图加一个，两个元路径子图）
        if len(edges) > 0:#如果存在边
            sorted_index = sorted(range(len(edges)), key=lambda i: edges[i])#按照第i个边进行排序，返回的应该是边列表长度对应的索引
            g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))#根据节点的索引向g中加边（还是两个图）
            result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
        else:#如果不存在边
            result_indices = torch.LongTensor(result_indices).to(device)
        # g.add_edges(*list(zip(*[(dst, src) for src, dst in sorted(edges)])))
        # result_indices = torch.LongTensor(result_indices).to(device)
        g_list.append(g.to(device))
        result_indices_list.append(result_indices)
        idx_batch_mapped_list.append(np.array([mapping[idx] for idx in idx_batch]))

    return g_list, result_indices_list, idx_batch_mapped_list#返回图列表（有两个子图），结果索引列表（也是有两个子图），batch的索引列表

#indices_list=[[1909,8459,500]...],type_mask=[],src_type:0,rate:0.3
def parse_mask(indices_list, type_mask, num_classes, src_type, rate, device):#进行特征遮蔽


    nodes = set()#nodes={1954,2338.。。。}
    for k in range(len(indices_list)):#遍历索引列表，也就是依次遍历每张图，k=0时i为17
        indices = indices_list[k].data.cpu().numpy()#将其转换为numpy格式
        for i in range(indices.shape[0]):#遍历行
            for j in range(indices.shape[1]):#遍历列,只有0,1,2
                nodes.add(indices[i, j])#节点集和增加第i行第j列
    nodes = [x for x in sorted(nodes)]#重新排序

    bound = [0]
    for i in range(num_classes):#遍历类型的数量
        bound.append(np.where(type_mask == i)[0][-1]+1)#bound添加第0行最后一列再加1？？？？？？？？？？

    mask_list = []
    for i in range(num_classes):#遍历类型的数量
        mask_list.append(np.array(sorted(search(nodes, bound[i], bound[i+1]-1))))#遮蔽列表，调用search函数

    feat_keep_idx, feat_drop_idx = train_test_split(np.arange(mask_list[src_type].shape[0]), test_size=rate)

    for i in range(num_classes):
        mask_list[i] = torch.LongTensor(mask_list[i]).to(device)
    feat_keep_idx = torch.LongTensor(feat_keep_idx).to(device)#特征保留的索引
    feat_drop_idx = torch.LongTensor(feat_drop_idx).to(device)#特征丢弃的索引
    return mask_list, feat_keep_idx, feat_drop_idx#返回遮蔽的列表，特征保留的索引，特征丢弃的索引


def search(lst, m, n):#这个函数是干什么用的？？？

    def search_upper_bound(lst, key):
        low = 0
        high = len(lst) - 1
        if key > lst[high]:
            return []
        if key <= lst[low]:
            return lst
        while low < high:
            mid = int((low + high+1) / 2)
            if lst[mid] < key:
                low = mid
            else:
                high = mid - 1
        if lst[low] <= key:
            return lst[low+1:]

    def search_lower_bound(lst, key):
        low = 0
        high = len(lst) - 1
        if key <= lst[low]:
            return []
        if key >= lst[high]:
            return lst
        while low < high:
            mid = int((low + high) / 2)
            if key < lst[mid]:
                high = mid
            else:
                low = mid + 1
        if key <= lst[low]:
            return lst[:low]
    return list(set(search_upper_bound(lst, m)) & set(search_lower_bound(lst, n)))


class index_generator:#索引生成
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:#这里跳过了
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:#这里是训练集和验证集还有测试集
            self.num_data = len(indices)#数量为训练集的数目
            self.indices = np.copy(indices)#深度拷贝，地址不一样
        self.batch_size = batch_size#batch_size设置为8
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)#如果为真的话，每次训练重新打乱顺序选取8个进行测试，（训练集进行了打乱，验证集和测试集不用）

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])#返回一个batch_size的索引

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))#向上取整，总数/batch_size=分成多少块

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter#就是看这一块更新完了吗。小于0就行更新完了，需要进行重新洗牌

    def reset(self):
        if self.shuffle:#如果重新洗牌了，则进行数据集洗牌
            np.random.shuffle(self.indices)
        self.iter_counter = 0#计数清零，这是iter_counter小于等于分成的块数
