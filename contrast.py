import torch
import torch.nn as nn
import numpy as np
import torch
from logreg import LogReg
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score
import torch
import scipy.sparse as sp
from tools import evaluate_results_nc
class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z_mg, z_sc, pos):
        z_proj_mp = self.proj(z_mg)
        z_proj_sc = self.proj(z_sc)
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = matrix_mp2sc.t()
        import torch

        # 假设 pos 是一个 Scipy COO 格式的稀疏矩阵
        coo = pos

        # 将 COO 格式稀疏矩阵转换为 PyTorch 稀疏张量
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.FloatTensor(coo.data)
        shape = torch.Size(coo.shape)
        sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)

        # 将稀疏张量转换为标准张量
        pos = sparse_tensor.to_dense()
        pos = pos.cuda()
        matrix_mp2sc = matrix_mp2sc / (torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        lori_mp = -torch.log(matrix_mp2sc.mul(pos.to_dense()).sum(dim=-1)).mean()

        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        lori_sc = -torch.log(matrix_sc2mp.mul(pos.to_dense()).sum(dim=-1)).mean()
        return self.lam * lori_mp + (1 - self.lam) * lori_sc  # 损失函数计算


##################################################
# This section of code adapted from pcy1302/DMGI #
##################################################

def evaluate(embeds, ratio,idx_train, idx_val, idx_test, label, nb_classes, device, dataset, lr, wd
             , isTest=True):
    hid_units = embeds.shape[1]
    xent = nn.CrossEntropyLoss()

    train_embs= embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]

    train_lbls = torch.argmax(label[idx_train], dim=-1)
    val_lbls = torch.argmax(label[idx_val], dim=-1)
    test_lbls = torch.argmax(label[idx_test], dim=-1)

    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []
    svm_macro_avg = np.zeros((7,), dtype=np.float)
    svm_micro_avg = np.zeros((7,), dtype=np.float)
    nmi_avg = 0
    ari_avg = 0
    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)
        log.to(device)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []

        logits_list = []
        for iter_ in range(200):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            logits_list.append(logits)
        svm_macro, svm_micro, nmi, ari = evaluate_results_nc(test_lbls.detach().cpu().numpy(),
                                                             preds.cpu().numpy(),
                                                             int(label.max()) + 1)  # 使用SVM评估节点
        svm_macro_avg = svm_macro_avg + svm_macro
        svm_micro_avg = svm_micro_avg + svm_micro
        nmi_avg += nmi
        ari_avg += ari
        # svm_macro_avg = svm_macro_avg / args['repeat']
        # svm_micro_avg = svm_micro_avg / args['repeat']
        # nmi_avg /= args['repeat']
        # ari_avg /= args['repeat']

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

        # auc
        best_logits = logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)
        auc_score_list.append(roc_auc_score(y_true=test_lbls.detach().cpu().numpy(),
                                            y_score=best_proba.detach().cpu().numpy(),
                                            multi_class='ovr'
                                            ))

    if isTest:

        print('Macro-F1: ' + ', '.join(['{:.6f}'.format(macro_f1) for macro_f1 in svm_macro_avg]))
        print('Micro-F1: ' + ', '.join(['{:.6f}'.format(micro_f1) for micro_f1 in svm_micro_avg]))
        print('NMI: {:.6f}'.format(nmi_avg))
        print('ARI: {:.6f}'.format(ari_avg))
        print('all finished')
    else:
        return np.mean(macro_f1s_val), np.mean(macro_f1s)


