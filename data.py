import numpy as np
import scipy
import pickle
import torch
import dgl
import torch.nn.functional as F
import torch as th
import scipy.sparse as sp

def load_ACM_data(prefix=r'D:\工作\基于正负样本距离的异质图神经网络\DATASET\ACM'):

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').toarray()

    # features_0 = np.eye(4019)
    # features_1 =np.eye(7167)
    # features_2 =np.eye(60)
    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)

    features_0 = F.normalize(features_0, dim=1, p=2)
    # features_1 = F.normalize(features_1, dim=1, p=2)
    # features_2 = F.normalize(features_2, dim=1, p=2)
    features = [features_0, features_1, features_2]
    pos = sp.load_npz(prefix + "/pos.npz")
    labels = np.load(prefix + '/labels.npy')#加载标签，4019
    labels = torch.LongTensor(labels)

    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')#加载训练集，验证集，测试集的索引

    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']
    type_mask = np.load(prefix + '/node_types.npy')
    num_classes = 3

    '''加载节点级信息'''

    ADJ = scipy.sparse.load_npz(prefix + '/adjM.npz').A
    PP = ADJ[0:4019, 0:4019]
    PP = torch.from_numpy(PP).type(torch.FloatTensor)
    PP = F.normalize(PP, dim=1, p=2)
    ADJ = ADJ[:4019, 4019:]
    ADJ = torch.FloatTensor(ADJ)
    ADJ = F.normalize(ADJ, dim=1, p=2)

    '''加载网络模式级信息'''
    nei_a = np.load(prefix + "/nei_a.npy", allow_pickle=True)
    nei_s = np.load(prefix + "/nei_s.npy", allow_pickle=True)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_s = [th.LongTensor(i) for i in nei_s]
    NS = [nei_a, nei_s]

    '''加载元图级信息'''
    PAP = scipy.sparse.load_npz(prefix + '/pap.npz').A
    PAP = torch.from_numpy(PAP).type(torch.FloatTensor)
    # PAP = PAP - torch.diag_embed(torch.diag(PAP))  # 去自环
    PAP = F.normalize(PAP, dim=1, p=2)#2范数按行归一化

    PSP = scipy.sparse.load_npz(prefix + '/psp.npz').A
    PSP = torch.from_numpy(PSP).type(torch.FloatTensor)
    # PSP = PSP - torch.diag_embed(torch.diag(PSP))  # 去自环
    PSP = F.normalize(PSP, dim=1, p=1)

    MG = [PAP, PSP,PP]  # 注意这里的元图有语义强度

    return ADJ, NS, MG, features, labels, num_classes, train_idx, val_idx, test_idx, type_mask,pos




def load_IMDB_data(prefix=r'D:\图神经网络学习\图神经网络学习\代码\异质图的多视图融合\异质图的多视图融合\DATASET\IMDB'):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').toarray()
    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)

    features_0 = F.normalize(features_0, dim=1, p=2)
    features_1 = F.normalize(features_1, dim=1, p=2)
    features_2 = F.normalize(features_2, dim=1, p=2)

    features = [features_0, features_1, features_2]

    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)
    # print(labels.shape)
    # zzz
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')  # 加载训练集，验证集，测试集的索引
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    type_mask = np.load(prefix + '/node_types.npy')
    num_classes = 3

    '''加载节点级信息'''
    ADJ = scipy.sparse.load_npz(prefix + '/adjM.npz').A
    ADJ = ADJ[:4278, 4278:]
    ADJ = torch.FloatTensor(ADJ)
    ADJ = F.normalize(ADJ, dim=1, p=1)

    '''加载网络模式级信息'''
    nei_a = np.load(prefix + "/nei_a.npy", allow_pickle=True)
    nei_d = np.load(prefix + "/nei_d.npy", allow_pickle=True)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_d = [th.LongTensor(i) for i in nei_d]
    NS = [nei_d, nei_a]


    '''加载元图级信息'''
    MDM = np.load(prefix + '/mdm.npy')
    MDM = torch.from_numpy(MDM).type(torch.FloatTensor)
    # MDM = MDM - torch.diag_embed(torch.diag(MDM))  # 去自环
    MDM = F.normalize(MDM, dim=1, p=2)

    MAM = np.load(prefix + '/mam.npy')
    MAM = torch.from_numpy(MAM).type(torch.FloatTensor)
    # MAM = MAM - torch.diag_embed(torch.diag(MAM))  # 去自环
    MAM = F.normalize(MAM, dim=1, p=2)

    MG = [MDM, MAM]  # 注意这里的图有语义强度
    pos = sp.load_npz(prefix + "\pos.npz")
    return ADJ, NS, MG, features, labels, num_classes, train_idx, val_idx, test_idx, type_mask




def load_DBLP_data(prefix=r'D:\图神经网络学习\图神经网络学习\代码\异质图的多视图融合\异质图的多视图融合\DATASET\DBLP'):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()#a
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()#p
    features_2 = np.load(prefix + '/features_2.npy')#t
    # features_0 = np.eye(4057)
    # features_1 = np.eye(14328)
    # features_2 = np.eye(7723)
    features_3 = np.eye(20)#c
    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)
    features_3 = torch.FloatTensor(features_3)
    pos = sp.load_npz(prefix + "\pos.npz")
    # features_0 = F.normalize(features_0, dim=1, p=0)
    #features_1 = F.normalize(features_1, dim=1, p=0)
    #features_2 = F.normalize(features_2, dim=1, p=0)

    features = [features_0, features_1, features_2, features_3]


    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)

    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    type_mask = np.load(prefix + '/node_types.npy')
    num_classes = 4

    '''加载节点级信息'''
    ADJ = scipy.sparse.load_npz(prefix + '/adjM.npz').A
    ADJ = ADJ[:4057, 4057:18385]
    ADJ = torch.FloatTensor(ADJ)
    ADJ = F.normalize(ADJ, dim=1, p=1)

    '''加载网络模式级信息 A-P'''
    nei_p = np.load(prefix + "/nei_p.npy", allow_pickle=True)
    nei_p = [th.LongTensor(i) for i in nei_p]
    # NS = [nei_p]

    '''加载网络模式级信息 P-C和P-T'''
    nei_c = np.load(prefix + "/nei_c.npy", allow_pickle=True)
    nei_c = [th.LongTensor(i) for i in nei_c]
    nei_t = np.load(prefix + "/nei_t.npy", allow_pickle=True)
    nei_t = [th.LongTensor(i) for i in nei_t]
    NS = [nei_p, nei_t, nei_c]

    '''加载元图级信息'''
    APA = scipy.sparse.load_npz(prefix + '/apa.npz').A
    APA = torch.from_numpy(APA).type(torch.FloatTensor)
    APA = APA - torch.diag_embed(torch.diag(APA))  # 去自环
    APA = F.normalize(APA, dim=1, p=1)

    APTPA = scipy.sparse.load_npz(prefix + '/aptpa.npz').A
    APTPA = torch.from_numpy(APTPA).type(torch.FloatTensor)
    APTPA = APTPA - torch.diag_embed(torch.diag(APTPA))  # 去自环
    APTPA = F.normalize(APTPA, dim=1, p=2)

    APCPA = scipy.sparse.load_npz(prefix + '/apvpa.npz').A
    APCPA = torch.from_numpy(APCPA).type(torch.FloatTensor)
    APCPA = APCPA - torch.diag_embed(torch.diag(APCPA))  # 去自环
    APCPA = F.normalize(APCPA, dim=1, p=2)

    MG = [APA, APTPA, APCPA]  # 注意这里的图有语义强度
    #MG = [APCPA]
    return ADJ, NS, MG, features, labels, num_classes, train_idx, val_idx, test_idx, type_mask,pos




def load_YELP_data(prefix=r'D:\图神经网络学习\图神经网络学习\代码\异质图的多视图融合\异质图的多视图融合\DATASET\YELP'):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0_b.npz').toarray()#b
    features_1 = scipy.sparse.load_npz(prefix + '/features_1_u.npz').toarray()#u
    features_2 = scipy.sparse.load_npz(prefix + '/features_2_s.npz').toarray()#s
    features_3 = scipy.sparse.load_npz(prefix + '/features_3_l.npz').toarray()#l
    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)
    features_3 = torch.FloatTensor(features_3)

    features_0 = F.normalize(features_0, dim=1, p=2)
    # features_1 = F.normalize(features_1, dim=1, p=1)
    # features_2 = F.normalize(features_2, dim=1, p=1)
    # features_3 = F.normalize(features_3, dim=1, p=1)

    features = [features_0, features_1, features_2, features_3]

    pos = sp.load_npz(prefix + "\pos.npz")
    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npy',allow_pickle=True)
    train_val_test_idx = train_val_test_idx.item()
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    type_mask = np.load(prefix + '/node_types.npy')
    num_classes = 3

    '''加载节点级信息'''
    ADJ = scipy.sparse.load_npz(prefix + '/adjM.npz').A

    ADJ = ADJ[:2614, 2614:]
    ADJ = torch.FloatTensor(ADJ)
    ADJ = F.normalize(ADJ, dim=1, p=0)

    '''加载网络模式级信息 '''
    nei_u = np.load(prefix + "/nei_u.npy", allow_pickle=True)
    # print(nei_u.shape)
    nei_u = [th.LongTensor(i) for i in nei_u]
    # NS = [nei_p]

    '''加载网络模式级信息 P-C和P-T'''
    nei_s = np.load(prefix + "/nei_s.npy", allow_pickle=True)
    nei_s = nei_s.astype(float)  # numpy强制类型转换

    nei_s = [th.LongTensor(i) for i in nei_s]
    nei_l = np.load(prefix + "/nei_l.npy", allow_pickle=True)
    nei_l = nei_l.astype(float)  # numpy强制类型转换
    nei_l = [th.LongTensor(i) for i in nei_l]
    NS = [nei_u, nei_s, nei_l]

    '''加载元图级信息'''
    BUB = scipy.sparse.load_npz(prefix + '/adj_bub.npz').A
    BUB = torch.from_numpy(BUB).type(torch.FloatTensor)
    #BUB = BUB - torch.diag_embed(torch.diag(BUB))  # 去自环
    BUB = F.normalize(BUB, dim=1, p=2)

    BSB = scipy.sparse.load_npz(prefix + '/adj_bsb.npz').A
    BSB = torch.from_numpy(BSB).type(torch.FloatTensor)
    #BSB = BSB - torch.diag_embed(torch.diag(BSB))  # 去自环
    BSB = F.normalize( BSB, dim=1, p=2)

    BLB = scipy.sparse.load_npz(prefix + '/adj_blb.npz').A
    BLB = torch.from_numpy(BLB).type(torch.FloatTensor)
    #BLB = BLB - torch.diag_embed(torch.diag(BLB))  # 去自环
    BLB = F.normalize(BLB, dim=1, p=2)

    MG = [BUB, BSB,BLB]  # 注意这里的图有语义强度

    return ADJ, NS, MG, features, labels, num_classes, train_idx, val_idx, test_idx, type_mask,pos
def load_FreeBase_data(prefix=r'D:\图神经网络学习\图神经网络学习\代码\异质图的多视图融合\DATASET\FreeBase'):

    features_0  = np.eye(3492)
    features_1 = np.eye(33401)
    features_2 = np.eye(2502)
    features_3 = np.eye(4459)
    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)
    features_3 = torch.FloatTensor(features_3)
    features_0 = F.normalize(features_0, dim=1, p=2)

    features = [features_0, features_1, features_2,features_3]

    labels = np.load(prefix + '/labels.npy')#加载标签，4019
    labels = torch.LongTensor(labels)

    train_idx = np.array(range(600))

    # val_idx = np.load(prefix+'lable_val.npy')
    val_idx = np.array(range(600, 900))

    # test_idx = np.load(prefix+'lable_test.npy')

    test_idx = np.array(range(900, 3492))

    type_mask = np.load(prefix + '/node_types.npy')
    num_classes = 4

    '''加载节点级信息'''
    ADJ = scipy.sparse.load_npz(prefix + '/adjM.npz').A
    ADJ = ADJ[:3492, 3492:]
    ADJ = torch.FloatTensor(ADJ)
    ADJ = F.normalize(ADJ, dim=1, p=1)

    '''加载网络模式级信息'''
    nei_a = np.load(prefix + "/nei_a.npy", allow_pickle=True)
    nei_d = np.load(prefix + "/nei_d.npy", allow_pickle=True)
    nei_w = np.load(prefix + "/nei_w.npy", allow_pickle=True)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_d = [th.LongTensor(i) for i in nei_d]
    nei_w = [th.LongTensor(i) for i in nei_w]
    NS = [nei_a, nei_d,nei_w]

    '''加载元图级信息'''
    MAM = scipy.sparse.load_npz(prefix + '/adj_mam.npz').A
    MAM = torch.from_numpy(MAM).type(torch.FloatTensor)
    #MAM = MAM - torch.diag_embed(torch.diag(MAM))  # 去自环
    MAM = F.normalize(MAM, dim=1, p=0)#2范数按行归一化

    MDM = scipy.sparse.load_npz(prefix + '/adj_mdm.npz').A
    MDM = torch.from_numpy(MDM).type(torch.FloatTensor)
    #MDM = MDM - torch.diag_embed(torch.diag(MDM))  # 去自环
    MDM = F.normalize(MDM, dim=1, p=2)
    MWM = scipy.sparse.load_npz(prefix + '/adj_mwm.npz').A
    MWM = torch.from_numpy(MWM).type(torch.FloatTensor)
    #MWM = MWM - torch.diag_embed(torch.diag(MWM))  # 去自环
    MWM = F.normalize(MWM, dim=1, p=2)
    pos = sp.load_npz(prefix + "\pos.npz")
    MG = [MAM,MDM,MWM]  # 注意这里的元图有语义强度

    return ADJ, NS, MG, features, labels, num_classes, train_idx, val_idx, test_idx, type_mask,pos
def load_DBLP_S_data(prefix=r'D:\图神经网络学习\图神经网络学习\代码\异质图的多视图融合\DATASET\DBLP_S'):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0_a.npz').toarray()#a
    features_1 = scipy.sparse.load_npz(prefix + '/features_1_p.npz').toarray()#p
    features_2 = scipy.sparse.load_npz(prefix + '/features_2_c.npz').toarray()#p
    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)

    features_0 = F.normalize(features_0, dim=1, p=2)
    # features_1 = F.normalize(features_1, dim=1, p=2)
    # features_2 = F.normalize(features_2, dim=1, p=2)

    features = [features_0, features_1, features_2]


    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)

    # train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    #train_idx = np.load(prefix+'lable_train.npy')
    train_idx = np.array(range(600))

    #val_idx = np.load(prefix+'lable_val.npy')
    val_idx= np.array(range(600,900))


    #test_idx = np.load(prefix+'lable_test.npy')

    test_idx = np.array(range(900,2957))

    # type_mask = np.load(prefix + '/node_types.npy')
    num_classes = 4

    '''加载节点级信息'''
    ADJ = scipy.sparse.load_npz(prefix + '/adjM.npz').A
    ADJ = ADJ[:2957, 2957:]
    #a=ADJ[:2957, 7285:]
    ADJ = torch.FloatTensor(ADJ)
    ADJ = F.normalize(ADJ, dim=1, p=1)

    '''加载网络模式级信息 A-P'''
    nei_p = np.load(prefix + "/nei_p.npy", allow_pickle=True)
    nei_p = [th.LongTensor(i) for i in nei_p]
    # NS = [nei_p]

    '''加载网络模式级信息 P-C和P-T'''
    nei_c = np.load(prefix + "/nei_c.npy", allow_pickle=True)
    nei_c = [th.LongTensor(i) for i in nei_c]
    # nei_t = np.load(prefix + "/nei_t.npy", allow_pickle=True)
    # nei_t = [th.LongTensor(i) for i in nei_t]
    NS = [nei_p, nei_c]

    '''加载元图级信息'''
    APA = scipy.sparse.load_npz(prefix + '/adj_apa.npz').A
    APA = torch.from_numpy(APA).type(torch.FloatTensor)
    APA = APA - torch.diag_embed(torch.diag(APA))  # 去自环
    APA = F.normalize(APA, dim=1, p=2)

    # APTPA = scipy.sparse.load_npz(prefix + '/aptpa.npz').A
    # APTPA = torch.from_numpy(APTPA).type(torch.FloatTensor)
    # APTPA = APTPA - torch.diag_embed(torch.diag(APTPA))  # 去自环
    # APTPA = F.normalize(APTPA, dim=1, p=2)

    APCPA = scipy.sparse.load_npz(prefix + '/adj_apcpa.npz').A
    APCPA = torch.from_numpy(APCPA).type(torch.FloatTensor)
    APCPA = APCPA - torch.diag_embed(torch.diag(APCPA))  # 去自环
    APCPA = F.normalize(APCPA, dim=1, p=2)
    pos = sp.load_npz(prefix + "\pos.npz")
    MG = [APA, APCPA]  # 注意这里的图有语义强度
    #MG = [APCPA]
    return ADJ, NS, MG, features, labels, num_classes, train_idx, val_idx, test_idx,pos
def load_Aminer_data(prefix=r'D:\图神经网络学习\图神经网络学习\代码\异质图的多视图融合\DATASET\Aminer'):

    features_0 = np.eye(6564)
    features_1 = np.eye(13329)
    features_2 = np.eye(35890)

    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)
    features_0 = F.normalize(features_0, dim=1, p=1)

    features = [features_0, features_1, features_2]#特征拼接
    labels = np.load(prefix + '/labels.npy')#加载标签，4019
    labels = torch.LongTensor(labels)
    pos = sp.load_npz(prefix + "\pos.npz")



    # type_mask = np.load(prefix + '/node_types.npy')

    train_idx = np.load(prefix + '/train_idx.npy')
    val_idx = np.load(prefix + '/val_idx.npy')
    test_idx = np.load(prefix + '/test_idx.npy')
    num_classes = 4

    '''加载节点级信息，节点的邻接矩阵'''
    ADJ = scipy.sparse.load_npz(prefix + '/adjm.npz').A

    ADJ = ADJ[:6564, 6564:]
    ADJ = torch.FloatTensor(ADJ)
    ADJ = F.normalize(ADJ, dim=1, p=1)

    '''加载网络模式级信息'''
    nei_a = np.load(prefix + "/nei_a.npy", allow_pickle=True)
    nei_r = np.load(prefix + "/nei_r.npy", allow_pickle=True)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_r = [th.LongTensor(i) for i in nei_r]

    NS = [nei_a, nei_r]

    '''加载元图级信息'''
    PAP = scipy.sparse.load_npz(prefix + '/adj_pap.npz').A
    PAP = torch.from_numpy(PAP).type(torch.FloatTensor)
    PAP = PAP - torch.diag_embed(torch.diag(PAP))  # 去自环
    PAP = F.normalize(PAP, dim=1, p=2)#2范数按行归一化

    PRP = scipy.sparse.load_npz(prefix + '/adj_prp.npz').A
    PRP = torch.from_numpy(PRP).type(torch.FloatTensor)
    PRP = PRP - torch.diag_embed(torch.diag(PRP))  # 去自环
    PRP = F.normalize(PRP, dim=1, p=2)


    MG = [PAP,PRP]  # 注意这里的元图有语义强
    return ADJ, NS, MG, features, labels, num_classes, train_idx, val_idx, test_idx,pos
def load_MAG_data(prefix=r'D:\图神经网络学习\图神经网络学习\代码\异质图的多视图融合\DATASET\MAG'):
    features_0 = np.load(prefix + '/features_p.npy')
    features_1 = np.load(prefix + '/features_a.npy')
    features_2 = np.load(prefix + '/features_i.npy')
    features_3 = np.load(prefix + '/features_f.npy')
    # features_0 = np.eye(len(features_0))
    # features_1 = np.eye(len(features_1))
    # features_2 = np.eye(len(features_2))
    # features_3 = np.eye(len(features_3))

    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)
    features_3 = torch.FloatTensor(features_3)
    # print(features_1.shape)
    # print(features_1.shape)
    # print(features_1.shape)
    # print(features_1.shape)
    # features_3 = F.normalize(features_3, dim=1, p=2)
    # features_0 = F.normalize(features_0, dim=1, p=2)
    # features_1 = F.normalize(features_1, dim=1, p=2)
    # features_2 = F.normalize(features_2, dim=1, p=2)
    features = [features_0, features_1, features_3,features_2]#特征拼接
    labels = np.load(prefix + '/p_label.npy')#加载标签，4019
    labels = torch.LongTensor(labels)

    train_idx = np.load(prefix + '/train_idx1.npy')
    # idx = np.random.permutation(4017)
    # train_idx = idx[:600]
    # val_idx = idx[600: 1200]
    #
    # test_idx = idx[1200: 4017]
    # # val_idx = np.load(prefix+'lable_val.npy')
    val_idx = np.load(prefix + '/val_idx1.npy')

    # test_idx = np.load(prefix+'lable_test.npy')

    test_idx = np.load(prefix + '/test_idx1.npy')
    # print(test_idx.shape,train_idx.shape,val_idx.shape)

    # type_mask = np.load(prefix + '/node_types.npy')

    # train_idx = np.load('DATASET/Aminer/train_idx.npy')
    # val_idx = np.load('DATASET/Aminer/val_idx.npy')
    # test_idx = np.load('DATASET/Aminer/test_idx.npy')
    num_classes = 4

    '''加载节点级信息，节点的邻接矩阵'''
    ADJ = scipy.sparse.load_npz(prefix + '/adjm.npz').A

    ADJ = ADJ[:4017, 4017:]
    ADJ = torch.FloatTensor(ADJ)
    ADJ = F.normalize(ADJ, dim=1, p=1)

    '''加载网络模式级信息'''
    nei_a = np.load(prefix + "/nei_a.npy", allow_pickle=True)
    # nei_i = np.load(prefix + "/nei_i.npy", allow_pickle=True)
    nei_f = np.load(prefix + "/nei_f.npy", allow_pickle=True)
    nei_a = [th.LongTensor(i) for i in nei_a]
    # nei_i = [th.LongTensor(i) for i in nei_i]
    nei_f = [th.LongTensor(i) for i in nei_f]

    NS = [nei_a,nei_f]
    #NS = [nei_f]

    '''加载元图级信息'''

    PAP = scipy.sparse.load_npz(prefix + '/adj_pap.npz').A
    PAP = torch.from_numpy(PAP).type(torch.FloatTensor)
    PAP = PAP - torch.diag_embed(torch.diag(PAP))  # 去自环
    PAP = F.normalize(PAP, dim=1, p=1)#2范数按行归一化

    PFP = scipy.sparse.load_npz(prefix + '/adj_pfp.npz').A
    PFP = torch.from_numpy(PFP).type(torch.FloatTensor)
    PFP = PFP - torch.diag_embed(torch.diag(PFP))  # 去自环
    PFP = F.normalize(PFP, dim=1, p=2)
    PAIAP = scipy.sparse.load_npz(prefix + '/adj_paiap.npz').A
    PAIAP = torch.from_numpy(PAIAP).type(torch.FloatTensor)
    PAIAP = PAIAP - torch.diag_embed(torch.diag(PAIAP))  # 去自环
    PAIAP = F.normalize(PAIAP, dim=1, p=2)
    pos = sp.load_npz(prefix + "\pos.npz")

    MG = [PAP,PFP,PAIAP]  # 注意这里的元图有语义强
    return ADJ, NS, MG, features, labels, num_classes, train_idx, val_idx, test_idx,pos