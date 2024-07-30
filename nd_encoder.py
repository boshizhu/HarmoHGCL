import torch
import torch.nn as nn
import torch.nn.functional as F

class Nd_encoder(nn.Module):
    def __init__(self, topo_dim, hidden_dim, feat_drop):
        super(Nd_encoder, self).__init__()
        self.hidden_dim = hidden_dim

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x

        self.feat_trans = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.topo_trans = nn.Linear(topo_dim, hidden_dim, bias=True)
        self.predict_1 = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)

    def forward(self, z_feat, ADJ):

        z_feat = F.elu(self.feat_drop(self.feat_trans(z_feat)))#拓扑信息和节点信息

        z_topo = F.elu(self.feat_drop(self.topo_trans(ADJ)))

        z = torch.cat((z_feat, z_topo), 1)
        # print(z.shape)
        h = F.elu(self.feat_drop(self.predict_1(z))) # 这里激活函数可以修改修改#imdb-elu
        # h=self.predict_1(z)
        # print(self.predict_1(z).shape)
        return z_feat,h
