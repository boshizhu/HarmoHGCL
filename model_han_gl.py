import torch
import torch.nn as nn
import torch.nn.functional as F
from model import HAN
import numpy as np

class HAN_GL(nn.Module):
    def __init__(self, input_dim, feat_hid_dim, metapath, dropout,outsize):
        super(HAN_GL,self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.metapath = metapath
        self.feat_hid_dim = feat_hid_dim
        self.non_linear = nn.ReLU()  # 定义非线性函数
        self.feat_mapping = nn.ModuleList([nn.Linear(m, feat_hid_dim, bias=True) for m in input_dim])  # 定义特征投影
        for fc in self.feat_mapping:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # 定义 语义特征图，语义结构图，总体新图
        self.overall_graph_gen = GraphChannelAttLayer(2,)

        self.het_graph_encoder_anchor = HAN(num_graph=2,  # 定义元路径长度（即确定多少个语义图）
                                            hidden_size=feat_hid_dim,  # 隐藏层维度
                                            out_size=outsize,  # 输出维度，3维
                                            num_layer=1,  # 网络层数
                                            dropout=dropout)


    def forward(self, features, G, ADJ, type_mask):
        # 特征映射
        transformed_features = torch.zeros(type_mask.shape[0], self.feat_hid_dim)
        for i, fc in enumerate(self.feat_mapping):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features[i])
        h = transformed_features
        feat_map = self.dropout(h)

        new_G = torch.zeros_like(G[0])  # 构建一个空的新图，有多少语义图生成多少个新图

        # 合并为新图
        #new_G = self.overall_graph_gen([G[0], G[1], G[2], G[3], G[4]])
        #new_G = self.overall_graph_gen([G[0], G[1], G[2], G[3]])
        #new_G = self.overall_graph_gen([G[0], G[1], G[2]])
        new_G = self.overall_graph_gen([G[0], G[1]])
        # new_G = torch.where(new_G < 0.2, torch.zeros_like(new_G), new_G)
        # 对称化
        # new_G = new_G.t() + new_G
        # 按照对列进行1范数归一化
        new_G = F.normalize(new_G, dim=1, p=1)#按行1范数归一化

        G = [ADJ, new_G]

        logits, h = self.het_graph_encoder_anchor(G, feat_map)
        return logits, h


# 图注意力层
class GraphChannelAttLayer(nn.Module):

    def __init__(self, num_channel, weights=None):
        super(GraphChannelAttLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel, 1, 1))
        nn.init.constant_(self.weight, 0.1)  # 初始化所有通道的权重相等
        if weights != None:  # 如果有自定义的权重
            self.weight.data = nn.Parameter(torch.Tensor(weights).reshape(self.weight.shape))
            #with torch.no_grad():
                #w = torch.Tensor(weights).reshape(self.weight.shape)
                #self.weight.copy_(w)  # 按照自定义的权重来

    def forward(self, adj_list):
        adj_list = torch.stack(adj_list)
        adj_list = F.normalize(adj_list, dim=1, p=1)  # 生成的图通过行归一化（1范数）

        #print(F.softmax(self.weight, dim=0))  # 这里可以打印每种图的注意力分配
        #a=torch.tensor([[[0.33333]],[[0.33333]]])
        return torch.sum(adj_list * F.softmax(self.weight, dim=0), dim=0)  # 每个图乘以它的注意系数
        #return torch.sum(adj_list * a, dim=0)