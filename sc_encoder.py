import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""类型间注意力"""
class inter_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(inter_att, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax(dim=0)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        # print("sc ", beta.data.cpu().numpy())  # 类型级注意力
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc

"""类型内注意力"""
class intra_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(intra_att, self).__init__()
        self.att = nn.Parameter(torch.empty(size=(1, 2*hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        self.softmax = nn.Softmax(dim=1)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, nei, h, h_refer):
        nei = nei.cuda()
        nei_emb = F.embedding(nei, h)

        '''均值聚合'''
        att = torch.ones([nei_emb.shape[0], nei_emb.shape[1], 1]).cuda()
        att = att/nei_emb.shape[1]

        '''注意力聚合'''
        # h_refer = torch.unsqueeze(h_refer, 1)
        # h_refer = h_refer.expand_as(nei_emb)
        # all_emb = torch.cat([h_refer, nei_emb], dim=-1)
        # attn_curr = self.attn_drop(self.att)
        # att = self.leakyrelu(all_emb.matmul(attn_curr.t()))
        # att = self.softmax(att)

        nei_emb = (att*nei_emb).sum(dim=1)

        return nei_emb


class Sc_encoder(nn.Module):
    def __init__(self, hidden_dim, sample_rate, nei_num, attn_drop):
        super(Sc_encoder, self).__init__()
        self.intra = nn.ModuleList([intra_att(hidden_dim, attn_drop) for _ in range(nei_num)])
        self.inter = inter_att(hidden_dim, attn_drop)
        self.sample_rate = sample_rate
        self.nei_num = nei_num
        self.feat_drop = nn.Dropout(attn_drop)
        self.predict_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, nei_h, nei_index):#h_all, NS
        embeds = []
        for i in range(self.nei_num):
            sele_nei = []
            sample_num = self.sample_rate[i]#采样邻居
            for per_node_nei in nei_index[i]:
                if len(per_node_nei) >= sample_num:
                    select_one = torch.tensor(np.random.choice(per_node_nei, sample_num,
                                                               replace=False))[np.newaxis]
                else:
                    select_one = torch.tensor(np.random.choice(per_node_nei, sample_num,
                                                               replace=True))[np.newaxis]
                sele_nei.append(select_one)
            sele_nei = torch.cat(sele_nei, dim=0)
            one_type_emb = (self.intra[i](sele_nei, nei_h[i + 1], nei_h[0]))#这里本来是i+1,DBLP
            #one_type_emb = self.feat_drop(one_type_emb)
            embeds.append(one_type_emb)
        # embeds.append(nei_h[0])
        z_mc = self.inter(embeds)#执行类型之间的注意力
        z_mc = F.tanh(self.feat_drop(self.predict_1(z_mc)))
        return z_mc
