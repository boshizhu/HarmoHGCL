import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from torch.nn.parameter import Parameter
import math

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_size, hidden_size),nn.Tanh(),nn.Linear(hidden_size, 1, bias=False))
    def forward(self, z):
        #w = self.project(z).mean(0)
        #beta = torch.softmax(w, dim=0)
        #print(beta)
        #beta=torch.tensor([[0.5],[0.5]]).to('cuda:0')
        #beta = beta.expand((z.shape[0],) + beta.shape)
        #return (beta * z).sum(1)

        beta = torch.Tensor([[1], [0]])
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)扩展到N个节点上的metapath的值
        z1 = (beta * z).sum(1)
        alpha = torch.Tensor([[0], [1]])
        alpha = alpha.expand((z.shape[0],) + alpha.shape)
        z2 = (alpha * z).sum(1)
        return torch.cat((z1,z2),1)


class GraphConvolution(nn.Module):  # 自己定义的GCN
    def __init__(self, in_features, out_features, bias=True):
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

    def forward(self, inputs, adj):
        support = torch.spmm(inputs, self.weight)  # HW in GCN
        output = torch.spmm(adj, support)  # AHW
        if self.bias is not None:
            return F.elu(output + self.bias)  # 这里激活函数按照定义为elu
        else:
            return F.elu(output)



class HANLayer(nn.Module):

    def __init__(self, num_graph, in_size, out_size, dropout):
        super(HANLayer, self).__init__()
        self.gcn_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        for i in range(num_graph):
            if i == 0:
                self.gcn_layers.append(GATConv(in_size, out_size, 1, dropout, dropout, activation=F.elu, allow_zero_in_degree=True))
            else:
                self.gcn_layers.append(GraphConvolution(in_size, out_size))
        self.semantic_attention = SemanticAttention(in_size=out_size)
        self.num_graph = num_graph

    def forward(self, gs, h):
        semantic_embeddings = []
        for i, g in enumerate(gs):
            if i == 0:
                #pass
                semantic_embeddings.append(self.gcn_layers[0](gs[0], h).flatten(1)[:len(gs[1]), :])
            else:
                #pass
                semantic_embeddings.append(self.gcn_layers[i](h[:len(g), :],g).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        return self.semantic_attention(semantic_embeddings)

class HAN(nn.Module):
    def __init__(self, num_graph, hidden_size, out_size, num_layer, dropout):
        super(HAN, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_graph, hidden_size, 32, dropout))
        for l in range(1, num_layer):
            self.layers.append(HANLayer(num_graph, hidden_size, hidden_size, dropout))
        self.predict = nn.Linear(32*2, out_size)

    def forward(self, g, h):
        a = 0  # DBLP中是0.2,ACM=0.1,IMDB=0
        h1 = h[:len(g[1]), :]
        for gnn in self.layers:
            h = gnn(g, h)
        #h = self.dropout(h)
        h = (1 - a) * h + a * h1
        return self.predict(h), h