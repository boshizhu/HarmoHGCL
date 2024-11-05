import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

class Mg_encoder(nn.Module):
    def __init__(self, hidden_dim, mp_len, dropout):
        super(Mg_encoder, self).__init__()

        self.non_linear = nn.ReLU()  # 定义非线性函数
        # 定义 语义特征图，语义结构图，总体新图
        self.overall_graph_gen = GraphChannelAttLayer(mp_len,)
        self.feat_drop = nn.Dropout(dropout)
        self.GCN = GraphConvolution(hidden_dim, hidden_dim)


    def forward(self, feature, MetaGraph, L_norm):#h_all[0], MS, L_norm

        new_G = self.overall_graph_gen(MetaGraph) # 合并为元图就是把元路径通过通道注意层连起来成为一个新的图
        # 按照对列进行1范数归一化
        new_G = F.normalize(new_G, dim=1, p=L_norm)  # IMDB进行2范数，ACM进行1范数
        feature = self.feat_drop(feature)
        h = F.tanh(self.GCN(feature, new_G))#yelp是elu
        # h = self.GCN(feature, new_G)
        return h,new_G


# 图注意力层
class GraphChannelAttLayer(nn.Module):

    def __init__(self, num_channel, weights=None):
        super(GraphChannelAttLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel, 1, 1))
        nn.init.constant_(self.weight, 0.1)  # 初始化所有通道的权重相等
        if weights != None:  # 如果有自定义的权重
            self.weight.data = nn.Parameter(torch.Tensor(weights).reshape(self.weight.shape))
            # with torch.no_grad():
                # w = torch.Tensor(weights).reshape(self.weight.shape)
                # self.weight.copy_(w)  # 按照自定义的权重来

    def forward(self, adj_list):#MS
        adj_list = torch.stack(adj_list)
        # adj_list = F.normalize(adj_list, dim=1, p=1)  # 生成的图通过行归一化（1范数）
        a = F.softmax(self.weight, dim=0)
        #print(a)
        return torch.sum(adj_list * a, dim=0)


# 自己定义的GCN
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):#参数是调用的里面的feature, new_G
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):  # 这里的权重和偏置归一化
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):#节点特征和邻接矩阵HA
        support = torch.spmm(inputs, self.weight)  # HW in GCN
        output = torch.spmm(adj, support)  # AHW
        if self.bias is not None:
            return output + self.bias  # 这里激活函数按照定义为elu
        else:
            return output