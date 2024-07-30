import torch
import torch.nn as nn
import torch.nn.functional as F
from mg_encoder import Mg_encoder
from sc_encoder import Sc_encoder
from nd_encoder import Nd_encoder
from contrast import Contrast
class MultiHGN(nn.Module):
    def __init__(self, topo_dim, hidden_dim, feats_dim_list, feat_drop, attn_drop, mp_num,num_classes):
        super(MultiHGN, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
                                      for feats_dim in feats_dim_list])

        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x

        self.nd = Nd_encoder(topo_dim, hidden_dim, feat_drop)#拓扑结构放到同一维度
        self.mg = Mg_encoder(hidden_dim, mp_num, feat_drop)
        self.predict_1 = nn.Linear(hidden_dim * 3, hidden_dim, bias=True)
        self.predict_2 = nn.Linear(hidden_dim * 1,num_classes, bias=True)
        self.predict_3 = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        #nn.init.xavier_normal_(self.predict_1.weight, gain=1.414)
        #nn.init.xavier_normal_(self.predict_2.weight, gain=1.414)
    def forward(self, features, ADJ, MetaGraph, L_norm):#features, ADJ, NS, MG, args['L_norm']
        h_all = []
        for i in range(len(features)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](features[i]))))
        h,z_nd = self.nd(h_all[0], ADJ)  # nd_encoder里面的前向传播
        z_mg,_ = self.mg(h_all[0], MetaGraph, L_norm)  # h

        return z_mg,h,z_nd
    def embed(self, features, ADJ, MetaGraph, L_norm):
        h_all = []
        for i in range(len(features)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](features[i]))))
        h,z_nd = self.nd(h_all[0], ADJ)  # nd_encoder里面的前向传播
        z_mg ,_= self.mg(h_all[0], MetaGraph, L_norm)  # h
        #z = torch.cat((h, z_nd), 1)#yelp
        z = torch.cat((h, z_mg), 1)  # yelp
        h_1 = F.elu((self.predict_3(z)))
        return h_1,self.predict_2(h_1)#yelp h_1