import argparse
import torch
from tools import evaluate_results_nc
from pytorchtools import EarlyStopping
from data import load_ACM_data,load_IMDB_data, load_DBLP_data, load_YELP_data
import numpy as np
import random
import time
from sklearn.metrics import f1_score
from mutiHGN import MultiHGN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import psutil
import os
import torch.nn.functional as F
import warnings
from torch_geometric.utils import degree
import torch.nn.functional as F
import warnings
from termcolor import cprint
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")
def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    return accuracy, micro_f1, macro_f1


'''剪枝操作'''
def edge_deletion(adj, drop_r):
    edge_index = np.array(np.nonzero(adj))
    half_edge_index = edge_index[:, edge_index[0,:] < edge_index[1,:]]
    num_edge = half_edge_index.shape[1]
    samples = np.random.choice(num_edge, size=int(drop_r * num_edge), replace=False)
    dropped_edge_index = half_edge_index[:, samples].T
    adj[dropped_edge_index[:,0],dropped_edge_index[:,1]] = 0.
    adj[dropped_edge_index[:,1],dropped_edge_index[:,0]] = 0.
    return adj

def main(args):
    ADJ, NS, MG, features, labels, num_classes, train_idx, val_idx, test_idx, type_mask,pos = load_YELP_data()
    in_dims = [feature.shape[1] for feature in features]  # 得到每种节点特征的维度
    topo_dim = ADJ.shape[1]  # 得到节点的拓扑信息
    ADJ = ADJ.to(args['device'])
    MG = [graph.to(args['device']) for graph in MG]
    features = [feature.to(args['device']) for feature in features]
    labels = labels.to(args['device'])
    nb_feature = features[0].shape[1]
    nb_nodes = features[0].shape[0]
    w_loss1 = args['w_loss1']
    w_loss2 = args['w_loss2']
    w_loss3 = args['w_loss3']
    svm_macro_avg = np.zeros((7,), dtype=np.float64)
    svm_micro_avg = np.zeros((7,), dtype=np.float64)
    nmi_avg = 0
    ari_avg = 0
    for cur_repeat in range(args['repeat']):
        set_random_seed(args['seed'] + cur_repeat)
        cprint("## Done ##", "yellow")
        model = MultiHGN(topo_dim, args['hidden_units'], in_dims, args['feat_drop'], args['att_drop'], len(MG),num_classes,
                        ).to(args['device'])
        early_stopping = EarlyStopping(patience=args['patience'], verbose=True,
                                       save_path='checkpoint/checkpoint_{}.pt'.format(args['dataset']))
        my_margin = args['margin1']  # 0.8
        my_margin_2 = my_margin + args['margin2']  # 0.2
        margin_loss = torch.nn.MarginRankingLoss(margin=my_margin, reduce=False)
        num_neg = 4  # 4
        lbl_z = torch.tensor([0.]).to(args['device'])
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        b = 0
        a = 0
        drop_rate = [0, 0]
        for epoch in range(args['num_epochs']):
            b = b + 1
            t = time.time()
            optimizer.zero_grad()
            model.train()
            idx_list = []  # 随机打乱
            for i in range(num_neg):
                idx_0 = np.random.permutation(nb_nodes)
                idx_list.append(idx_0)
            h_a, h_p, h_p_1 = model(features, ADJ, MG, args['L_norm'])
            s_p = F.pairwise_distance(h_a, h_p)
            s_p_1 = F.pairwise_distance(h_a, h_p_1)
            s_n_list = []
            for h_n in idx_list:
                s_n = F.pairwise_distance(h_a, h_a[h_n])
                s_n_list.append(s_n)
            margin_label = -1 * torch.ones_like(s_p)

            loss_mar = 0
            loss_mar_1 = 0
            mask_margin_N = 0
            for s_n in s_n_list:
                loss_mar += (margin_loss(s_p, s_n, margin_label)).mean()
                loss_mar_1 += (margin_loss(s_p_1, s_n, margin_label)).mean()
                mask_margin_N += torch.max((s_n - s_p.detach() - my_margin_2), lbl_z).sum()
            mask_margin_N = mask_margin_N / num_neg

            loss = loss_mar * w_loss1 + loss_mar_1 * w_loss2 + mask_margin_N * w_loss3  # w1Ls +w2Ln+Lu
            print('Epoch{:d}| Loss{:.4f}'.format(epoch + 1, loss.item()))
            print(u'当前进程的内存使用:%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
            early_stopping(loss.data.item(), model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break
            loss.backward()
            optimizer.step()
            t2 = time.time()
            a = a + (t2 - t)
        # print(u'当前进程的内存使用:%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        model.eval()
        print('平均时间:', a / b)
        model.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(args['dataset'])))
        h,logits = model.embed(features, ADJ, MG, args['L_norm'])
        svm_macro, svm_micro, nmi, ari = evaluate_results_nc(h[test_idx].detach().cpu().numpy(), labels[test_idx].cpu().numpy(),
                                                             int(labels.max()) + 1)  # 使用SVM评估节点
        svm_macro_avg = svm_macro_avg + svm_macro
        svm_micro_avg = svm_micro_avg + svm_micro
        nmi_avg += nmi
        ari_avg += ari
        with open('results_yelp.txt', 'a') as file:
            file.write(str(args['dataset']) + '\n')
            file.write(str('w1') + str(':') + str(args['margin1']) + '\n')
            file.write(str('w2') + str(':') + str(args['margin2']) + '\n')
            # file.write(str('w2') + str(':') + str(args['w_ToPo']) + '\n')
            file.write(str('Macro-F1:') + str(svm_macro_avg[0]) + '\n')
            # file.write(str('Micro-F1:') + str(svm_micro_avg) + '\n')
            file.write(str('NMI:') + str(nmi_avg) + '\n')
            file.write(str('ARI:') + str(ari_avg) + '\n')
        # ======================#这个作用是可视化
        # Y = labels[test_idx].cpu().numpy()
        # ml = TSNE(n_components=2)
        # node_pos = ml.fit_transform(h[test_idx].detach().cpu().numpy())
        # color_idx = {}
        # for i in range(len(h[test_idx].detach().cpu().numpy())):
        #     color_idx.setdefault(Y[i], [])
        #     color_idx[Y[i]].append(i)
        # for c, idx in color_idx.items():  # c是类型数，idx是索引
        #     if str(c) == '1':
        #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#DAA520', s=15, alpha=1)
        #     elif str(c) == '2':
        #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#8B0000', s=15, alpha=1)
        #     elif str(c) == '0':
        #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#6A5ACD', s=15, alpha=1)
        #     elif str(c) == '3':
        #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#006400', s=15, alpha=1)
        # plt.legend()
        # plt.savefig(".\\visualization\OSGNN_323_" + str(args['dataset']) + "分类图" + str(cur_repeat) + ".png", dpi=1000,
        # 			bbox_inches='tight')
        # plt.show()

    svm_macro_avg = svm_macro_avg / args['repeat']
    svm_micro_avg = svm_micro_avg / args['repeat']
    nmi_avg /= args['repeat']
    ari_avg /= args['repeat']
    print('---\nThe average of {} results:'.format(args['repeat']))
    print('Macro-F1: ' + ', '.join(['{:.6f}'.format(macro_f1) for macro_f1 in svm_macro_avg]))
    print('Micro-F1: ' + ', '.join(['{:.6f}'.format(micro_f1) for micro_f1 in svm_micro_avg]))
    print('NMI: {:.6f}'.format(nmi_avg))
    print('ARI: {:.6f}'.format(ari_avg))
    print('all finished')


if __name__ == '__main__':
    for i in range(1):
        for j in range(1):
            print("当前第", str(i + j), "次==========================================================================")
            parser = argparse.ArgumentParser(description='这是我们多级聚合模型')
            parser.add_argument('--dataset', default='YELP', help='数据集')
            parser.add_argument('--lr', default=0.002, help='学习率') #imdb=0.002 #ACM=0.002 #DBLP=0.001
            parser.add_argument('--weight_decay', default=0.000, help='权重衰减') #imdb=0.000 #ACM=0.0014
            parser.add_argument('--hidden_units', default=64, help='隐藏层数')
            parser.add_argument('--att_drop', default=0.5, help='注意力丢弃率')
            parser.add_argument('--feat_drop', default=0.5, help='特征丢弃率')
            parser.add_argument('--L_norm', default=2, help='归一化类型')
            parser.add_argument('--num_epochs', default=1000, help='最大迭代次数')
            parser.add_argument('--patience', type=int, default=20, help='耐心值')
            parser.add_argument('--seed', type=int, default=164, help='随机种子')#imdb=164  #ACM=164  #DBLP=164
            parser.add_argument('--device', type=str, default='cuda:0', help='使用cuda:0或者cpu')
            parser.add_argument('--repeat', type=int, default=1, help='重复训练和测试次数')

            # parser.add_argument('--margin1', type=float, default=0 + 0.1 * i, help='')  # 0.8
            # parser.add_argument('--margin2', type=float, default=0 + 0.1 * j, help='')  # 0.2
            parser.add_argument('--margin1', type=float, default=1, help='')
            parser.add_argument('--margin2', type=float, default=0.2, help='')

            parser.add_argument('--w_loss1', type=float, default=10, help='')#10
            parser.add_argument('--w_loss2', type=float, default=10, help='')
            # parser.add_argument('--w_loss1', type=float, default=0.001 * (10 ** i), help='')  # 60高阶异质信息
            # parser.add_argument('--w_loss2', type=float, default=0.001 * (10 ** i), help='')  # 60局部信息
            parser.add_argument('--w_loss3', type=float, default=1, help='')
            parser.add_argument('--way', default='noself')
            args = parser.parse_args().__dict__
            set_random_seed(args['seed'])
            print(args)
            main(args)

