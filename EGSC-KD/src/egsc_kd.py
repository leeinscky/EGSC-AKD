import torch
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
from scipy.stats import spearmanr, kendalltau

from layers import AttentionModule, TensorNetworkModule, DiffPool
from utils import calculate_ranking_correlation, calculate_prec_at_k, gen_pairs

from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.data import DataLoader, Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import OneHotDegree

import matplotlib.pyplot as plt

from model_kd import EGSC_generator, EGSC_fusion, EGSC_fusion_classifier, EGSC_classifier, EGSC_teacher, logits_D

import pdb
from layers import RkdDistance, RKdAngle, repeat_certain_graph

class EGSC_KD_Trainer(object):
    def __init__(self, args):
        self.args = args
        self.process_dataset()
        self.setup_model()
        self.best_rho = 0
        self.best_tau = 0
        self.best_prec_at_10 = 0
        self.best_prec_at_20 = 0
        self.best_model_error = float('inf')

    def setup_model(self):
        self.model_g = EGSC_generator(self.args, self.number_of_labels)
        self.model_f = EGSC_fusion(self.args, self.number_of_labels)
        self.model_c = EGSC_classifier(self.args, self.number_of_labels)
        self.model_c1 = EGSC_fusion_classifier(self.args, self.number_of_labels)
        self.model_g_fix = EGSC_teacher(self.args, self.number_of_labels) # number_of_labels = 29
        self.Discriminator = logits_D(16, 16)
        """ print(f'[EGSC-KD/src/egsc_kd.py] self.model_g: {self.model_g}')
        print(f'[EGSC-KD/src/egsc_kd.py] self.model_f: {self.model_f}')
        print(f'[EGSC-KD/src/egsc_kd.py] self.model_c: {self.model_c}')
        print(f'[EGSC-KD/src/egsc_kd.py] self.model_c1: {self.model_c1}')
        print(f'[EGSC-KD/src/egsc_kd.py] self.model_g_fix: {self.model_g_fix}') """
        
        """ --dataset AIDS700nef 打印结果
        
        [EGSC-KD/src/egsc_kd.py] self.model_g: EGSC_generator(
        (convolution_1): GINConv(nn=Sequential(
            (0): Linear(in_features=29, out_features=64, bias=True)
            (1): ReLU()
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ))
        (convolution_2): GINConv(nn=Sequential(
            (0): Linear(in_features=64, out_features=32, bias=True)
            (1): ReLU()
            (2): Linear(in_features=32, out_features=32, bias=True)
            (3): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ))
        (convolution_3): GINConv(nn=Sequential(
            (0): Linear(in_features=32, out_features=16, bias=True)
            (1): ReLU()
            (2): Linear(in_features=16, out_features=16, bias=True)
            (3): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ))
        (attention): AttentionModule()
        (attention_level2): AttentionModule()
        (attention_level1): AttentionModule()
        )
        
        [EGSC-KD/src/egsc_kd.py] self.model_f: EGSC_fusion(
        (feat_layer): Linear(in_features=224, out_features=112, bias=True)
        (fully_connected_first): Linear(in_features=112, out_features=16, bias=True)
        (score_attention): SEAttentionModule(
            (fc): Sequential(
            (0): Linear(in_features=224, out_features=56, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=56, out_features=224, bias=True)
            (3): Sigmoid()
            )
        )
        )
        
        [EGSC-KD/src/egsc_kd.py] self.model_c: EGSC_classifier(
        (scoring_layer): Linear(in_features=16, out_features=1, bias=True)
        )
        
        [EGSC-KD/src/egsc_kd.py] self.model_c1: EGSC_fusion_classifier(
        (feat_layer): Linear(in_features=32, out_features=16, bias=True)
        (scoring_layer): Linear(in_features=16, out_features=1, bias=True)
        )
        
        [EGSC-KD/src/egsc_kd.py] self.model_g_fix: EGSC_teacher(
        (convolution_1): GINConv(nn=Sequential(
            (0): Linear(in_features=29, out_features=64, bias=True)
            (1): ReLU()
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ))
        (convolution_2): GINConv(nn=Sequential(
            (0): Linear(in_features=64, out_features=32, bias=True)
            (1): ReLU()
            (2): Linear(in_features=32, out_features=32, bias=True)
            (3): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ))
        (convolution_3): GINConv(nn=Sequential(
            (0): Linear(in_features=32, out_features=16, bias=True)
            (1): ReLU()
            (2): Linear(in_features=16, out_features=16, bias=True)
            (3): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ))
        (attention_level3): AttentionModule_fix(
            (fc): Sequential(
            (0): Linear(in_features=16, out_features=4, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=4, out_features=16, bias=True)
            (3): Tanh()
            )
            (fc1): Linear(in_features=16, out_features=16, bias=True)
        )
        (attention_level2): AttentionModule_fix(
            (fc): Sequential(
            (0): Linear(in_features=32, out_features=8, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=8, out_features=32, bias=True)
            (3): Tanh()
            )
            (fc1): Linear(in_features=32, out_features=32, bias=True)
        )
        (attention_level1): AttentionModule_fix(
            (fc): Sequential(
            (0): Linear(in_features=64, out_features=16, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=16, out_features=64, bias=True)
            (3): Tanh()
            )
            (fc1): Linear(in_features=64, out_features=64, bias=True)
        )
        (cross_attention_level2): CrossAttentionModule(
            (multihead_attention): MultiHeadAttention(
            (w_qs): Linear(in_features=32, out_features=32, bias=False)
            (w_ks): Linear(in_features=32, out_features=32, bias=False)
            (w_vs): Linear(in_features=32, out_features=32, bias=False)
            (fc): Linear(in_features=32, out_features=32, bias=False)
            (attention): ScaledDotProductAttention(
                (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
            (layer_norm): LayerNorm((32,), eps=1e-06, elementwise_affine=True)
            )
            (mlp): PositionwiseFeedForward(
            (w_1): Linear(in_features=32, out_features=32, bias=True)
            (w_2): Linear(in_features=32, out_features=32, bias=True)
            (layer_norm): LayerNorm((32,), eps=1e-06, elementwise_affine=True)
            (batch_norm): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (dropout): Dropout(p=0.1, inplace=False)
            )
        )
        (cross_attention_level3): CrossAttentionModule(
            (multihead_attention): MultiHeadAttention(
            (w_qs): Linear(in_features=16, out_features=16, bias=False)
            (w_ks): Linear(in_features=16, out_features=16, bias=False)
            (w_vs): Linear(in_features=16, out_features=16, bias=False)
            (fc): Linear(in_features=16, out_features=16, bias=False)
            (attention): ScaledDotProductAttention(
                (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
            (layer_norm): LayerNorm((16,), eps=1e-06, elementwise_affine=True)
            )
            (mlp): PositionwiseFeedForward(
            (w_1): Linear(in_features=16, out_features=16, bias=True)
            (w_2): Linear(in_features=16, out_features=16, bias=True)
            (layer_norm): LayerNorm((16,), eps=1e-06, elementwise_affine=True)
            (batch_norm): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (dropout): Dropout(p=0.1, inplace=False)
            )
        )
        (cross_attention_level4): CrossAttentionModule(
            (multihead_attention): MultiHeadAttention(
            (w_qs): Linear(in_features=8, out_features=8, bias=False)
            (w_ks): Linear(in_features=8, out_features=8, bias=False)
            (w_vs): Linear(in_features=8, out_features=8, bias=False)
            (fc): Linear(in_features=8, out_features=8, bias=False)
            (attention): ScaledDotProductAttention(
                (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
            (layer_norm): LayerNorm((8,), eps=1e-06, elementwise_affine=True)
            )
            (mlp): PositionwiseFeedForward(
            (w_1): Linear(in_features=8, out_features=8, bias=True)
            (w_2): Linear(in_features=8, out_features=8, bias=True)
            (layer_norm): LayerNorm((8,), eps=1e-06, elementwise_affine=True)
            (batch_norm): BatchNorm1d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (dropout): Dropout(p=0.1, inplace=False)
            )
        )
        (tensor_network_level3): SETensorNetworkModule(
            (fc_se): Sequential(
            (0): Linear(in_features=32, out_features=8, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=8, out_features=32, bias=True)
            (3): Sigmoid()
            )
            (fc0): Sequential(
            (0): Linear(in_features=32, out_features=32, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=32, out_features=32, bias=True)
            (3): ReLU(inplace=True)
            )
            (fc1): Sequential(
            (0): Linear(in_features=32, out_features=32, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=32, out_features=8, bias=True)
            (3): ReLU(inplace=True)
            )
        )
        (tensor_network_level2): SETensorNetworkModule(
            (fc_se): Sequential(
            (0): Linear(in_features=64, out_features=16, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=16, out_features=64, bias=True)
            (3): Sigmoid()
            )
            (fc0): Sequential(
            (0): Linear(in_features=64, out_features=64, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): ReLU(inplace=True)
            )
            (fc1): Sequential(
            (0): Linear(in_features=64, out_features=64, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=64, out_features=16, bias=True)
            (3): ReLU(inplace=True)
            )
        )
        (tensor_network_level1): SETensorNetworkModule(
            (fc_se): Sequential(
            (0): Linear(in_features=128, out_features=32, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=32, out_features=128, bias=True)
            (3): Sigmoid()
            )
            (fc0): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=128, out_features=128, bias=True)
            (3): ReLU(inplace=True)
            )
            (fc1): Sequential(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=128, out_features=32, bias=True)
            (3): ReLU(inplace=True)
            )
        )
        (fully_connected_first): Linear(in_features=56, out_features=16, bias=True)
        (scoring_layer): Linear(in_features=16, out_features=1, bias=True)
        (score_attention): SEAttentionModule(
            (fc): Sequential(
            (0): Linear(in_features=56, out_features=14, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=14, out_features=56, bias=True)
            (3): Sigmoid()
            )
        )
        ) """



        self.loss_RkdDistance = RkdDistance()
        self.loss_RKdAngle = RKdAngle()

    # def save_model(self):
    #     PATH_g = '../model_sel/G_' +str(self.args.dataset)+"_"+ str(round(self.model_error*1000, 5))+"_" \
    #     + str(self.args.epochs)+"_"+str(self.args.batch_size)+"_"+str(self.args.learning_rate) +'_checkpoint.pth'
    #     PATH_c = '../model_sel/C_' +str(self.args.dataset)+"_"+ str(round(self.model_error*1000, 5))+"_" \
    #     + str(self.args.epochs)+"_"+str(self.args.batch_size)+"_"+str(self.args.learning_rate) +'_checkpoint.pth'
        
    #     torch.save(self.model_g.state_dict(), PATH_g)
    #     torch.save(self.model_c.state_dict(), PATH_c)
        
    #     print('Model Saved')

    def load_model(self):
        PATH_g = '../Checkpoints/G_EarlyFusion_Disentangle_' +str(self.args.dataset) +'_gin'+'_checkpoint.pth'

        self.model_g_fix.load_state_dict(torch.load(PATH_g)) # load_state_dict使用 state_dict 反序列化模型参数字典。用来加载模型参数。将 state_dict 中的 parameters 和 buffers 复制到此 module 及其子节点中。 概况：给模型对象加载训练好的模型参数，即加载模型参数

        print('Model Loaded')
        
    def process_dataset(self):
        print("\nPreparing dataset.\n")

        self.args.data_dir = '../GSC_datasets'

        self.training_graphs = GEDDataset(self.args.data_dir+'/{}'.format(self.args.dataset), self.args.dataset, train=True) 
        self.testing_graphs = GEDDataset(self.args.data_dir+'/{}'.format(self.args.dataset), self.args.dataset, train=False) 
        if self.args.dataset=="ALKANE":
            self.testing_graphs = GEDDataset(self.args.data_dir+'/{}'.format(self.args.dataset), self.args.dataset, train=True) 
        self.nged_matrix = self.training_graphs.norm_ged
        self.ged_matrix = self.training_graphs.ged

        self.real_data_size = self.nged_matrix.size(0)
        
        if self.args.synth:
            self.synth_data_1, self.synth_data_2, _, synth_nged_matrix = gen_pairs(self.training_graphs.shuffle()[:500], 0, 3)  
            
            real_data_size = self.nged_matrix.size(0)
            synth_data_size = synth_nged_matrix.size(0)
            self.nged_matrix = torch.cat((self.nged_matrix, torch.full((real_data_size, synth_data_size), float('inf'))), dim=1)
            synth_nged_matrix = torch.cat((torch.full((synth_data_size, real_data_size), float('inf')), synth_nged_matrix), dim=1)
            self.nged_matrix = torch.cat((self.nged_matrix, synth_nged_matrix))
        
        if self.training_graphs[0].x is None:
            max_degree = 0
            for g in self.training_graphs + self.testing_graphs + (self.synth_data_1 + self.synth_data_2 if self.args.synth else []):
                if g.edge_index.size(1) > 0:
                    max_degree = max(max_degree, int(degree(g.edge_index[0]).max().item()))
            one_hot_degree = OneHotDegree(max_degree, cat=False)
            self.training_graphs.transform = one_hot_degree
            self.testing_graphs.transform = one_hot_degree
        
        # labeling of synth data according to real data format    
            if self.args.synth:
                for g in self.synth_data_1 + self.synth_data_2:
                    g = one_hot_degree(g)
                    g.i = g.i + real_data_size
        elif self.args.synth:
            for g in self.synth_data_1 + self.synth_data_2:
                g.i = g.i + real_data_size    
        self.number_of_labels = self.training_graphs.num_features

    def create_batches(self):
        if self.args.synth:
            synth_data_ind = random.sample(range(len(self.synth_data_1)), 100)
        
        source_loader = DataLoader(self.training_graphs.shuffle() + 
            ([self.synth_data_1[i] for i in synth_data_ind] if self.args.synth else []), batch_size=self.args.batch_size)
        target_loader = DataLoader(self.training_graphs.shuffle() + 
            ([self.synth_data_2[i] for i in synth_data_ind] if self.args.synth else []), batch_size=self.args.batch_size)
        
        return list(zip(source_loader, target_loader))

    def transform(self, data):
        new_data = dict()

        new_data["g1"] = data[0]
        new_data["g2"] = data[1]
        normalized_ged = self.nged_matrix[data[0]["i"].reshape(-1).tolist(),data[1]["i"].reshape(-1).tolist()].tolist()
        new_data["target"] = torch.from_numpy(np.exp([(-el) for el in normalized_ged])).view(-1).float()
        ged = self.ged_matrix[data[0]["i"].reshape(-1).tolist(),data[1]["i"].reshape(-1).tolist()].tolist()
        new_data["target_ged"] = torch.from_numpy(np.array([(el) for el in ged])).view(-1).float()

        return new_data

    def process_batch(self, data):
        self.optimizer_g.zero_grad()
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        self.optimizer_c1.zero_grad()
        
        data = self.transform(data)
        target = data["target"]

        edge_index_1 = data["g1"].edge_index
        edge_index_2 = data["g2"].edge_index
        features_1 = data["g1"].x
        features_2 = data["g2"].x
        batch_1 = data["g1"].batch if hasattr(data["g1"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes)
        batch_2 = data["g2"].batch if hasattr(data["g2"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes)

        pooled_features_1_all = self.model_g(edge_index_1, features_1, batch_1)
        pooled_features_2_all = self.model_g(edge_index_2, features_2, batch_2)

        prediction = self.model_c(self.model_f(pooled_features_1_all, pooled_features_2_all))
        loss_reg = F.mse_loss(prediction, target, reduction='sum') #* 0.5

        pooled_features_1 = self.model_g(edge_index_1, features_1, batch_1)
        pooled_features_2 = self.model_g(edge_index_2, features_2, batch_2)

        feat_joint = self.model_f(pooled_features_1, pooled_features_2)
        feat_joint_1 = self.model_f(pooled_features_1, pooled_features_1)
        feat_joint_2 = self.model_f(pooled_features_2, pooled_features_2)

        feat_joint_fix = self.model_g_fix(edge_index_1, features_1, batch_1, edge_index_2, features_2, batch_2).detach()
        feat_joint_fix_1 = self.model_g_fix(edge_index_1, features_1, batch_1, edge_index_1, features_1, batch_1).detach()
        feat_joint_fix_2 = self.model_g_fix(edge_index_2, features_2, batch_2, edge_index_2, features_2, batch_2).detach()
        feat_1 = feat_joint-feat_joint_1
        feat_fix_1 = feat_joint_fix-feat_joint_fix_1
        feat_2 = feat_joint-feat_joint_2
        feat_fix_2 = feat_joint_fix-feat_joint_fix_2

        if self.args.mode == "l1":
            loss_kd = (F.smooth_l1_loss(feat_1, feat_fix_1) + \
            F.smooth_l1_loss(feat_2, feat_fix_2) ) * 10

        elif self.args.mode == "rkd_dis":
            loss_kd = (self.loss_RkdDistance(feat_1, feat_fix_1) + \
            self.loss_RkdDistance(feat_2, feat_fix_2)) * 10
        elif self.args.mode == "rkd_ang":
            loss_kd = (self.loss_RKdAngle(feat_1, feat_fix_1) + \
            self.loss_RKdAngle(feat_2, feat_fix_2)) * 10

        elif self.args.mode == "both":
            loss_kd = (F.smooth_l1_loss(feat_1, feat_fix_1) + \
            F.smooth_l1_loss(feat_2, feat_fix_2) ) * 5 + \
            (self.loss_RkdDistance(feat_1, feat_fix_1) + \
            self.loss_RkdDistance(feat_2, feat_fix_2)) * 5

        else:
            loss_kd = 0

        feat_12 = torch.cat((feat_1, feat_2),dim=1)
        prediction_rec = self.model_c1(feat_12)

        loss_reg_rec = F.mse_loss(prediction, target, reduction='sum') # Apart from the KD loss, there is a supervision (i.e., MSE) loss LSreg on the student model to fulfill the object of the task.

        loss = loss_reg + loss_kd + loss_reg_rec # loss_reg是loss_regression loss_kd是loss_knowledge_distillation
        loss.backward()
        self.optimizer_c1.step()
        self.optimizer_c.step()
        self.optimizer_f.step()
        self.optimizer_g.step()
        return loss_reg.item(), loss_kd.item()
    
    def process_batch_adversarial(self, data, epoch):
        self.optimizer_g.zero_grad()
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        self.optimizer_c1.zero_grad()
        self.opt_D.zero_grad()
        
        data = self.transform(data)
        target = data["target"]

        edge_index_1 = data["g1"].edge_index
        edge_index_2 = data["g2"].edge_index
        features_1 = data["g1"].x
        features_2 = data["g2"].x
        batch_1 = data["g1"].batch if hasattr(data["g1"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes)
        batch_2 = data["g2"].batch if hasattr(data["g2"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes)

        pooled_features_1_all = self.model_g(edge_index_1, features_1, batch_1)
        pooled_features_2_all = self.model_g(edge_index_2, features_2, batch_2)

        prediction = self.model_c(self.model_f(pooled_features_1_all, pooled_features_2_all))
        loss_reg = F.mse_loss(prediction, target, reduction='sum') #* 0.5
        # print(f'[EGSC-KD/src/egsc_kd.py] 正在执行process_batch_adversarial函数，prediction.shape={prediction.shape}, target.shape={target.shape}') # prediction.shape=torch.Size([128]), target.shape=torch.Size([128])
        # label_loss = self.loss_fcn(prediction, target)
        label_loss = loss_reg

        pooled_features_1 = self.model_g(edge_index_1, features_1, batch_1)
        pooled_features_2 = self.model_g(edge_index_2, features_2, batch_2)

        feat_joint = self.model_f(pooled_features_1, pooled_features_2)
        feat_joint_1 = self.model_f(pooled_features_1, pooled_features_1)
        feat_joint_2 = self.model_f(pooled_features_2, pooled_features_2)

        feat_joint_fix = self.model_g_fix(edge_index_1, features_1, batch_1, edge_index_2, features_2, batch_2).detach()
        feat_joint_fix_1 = self.model_g_fix(edge_index_1, features_1, batch_1, edge_index_1, features_1, batch_1).detach()
        feat_joint_fix_2 = self.model_g_fix(edge_index_2, features_2, batch_2, edge_index_2, features_2, batch_2).detach()
        feat_1 = feat_joint-feat_joint_1
        feat_fix_1 = feat_joint_fix-feat_joint_fix_1
        feat_2 = feat_joint-feat_joint_2
        feat_fix_2 = feat_joint_fix-feat_joint_fix_2

        if self.args.mode == "l1":
            loss_kd = (F.smooth_l1_loss(feat_1, feat_fix_1) + \
            F.smooth_l1_loss(feat_2, feat_fix_2) ) * 10

        elif self.args.mode == "rkd_dis":
            loss_kd = (self.loss_RkdDistance(feat_1, feat_fix_1) + \
            self.loss_RkdDistance(feat_2, feat_fix_2)) * 10
        elif self.args.mode == "rkd_ang":
            loss_kd = (self.loss_RKdAngle(feat_1, feat_fix_1) + \
            self.loss_RKdAngle(feat_2, feat_fix_2)) * 10

        elif self.args.mode == "both":
            loss_kd = (F.smooth_l1_loss(feat_1, feat_fix_1) + \
            F.smooth_l1_loss(feat_2, feat_fix_2) ) * 5 + \
            (self.loss_RkdDistance(feat_1, feat_fix_1) + \
            self.loss_RkdDistance(feat_2, feat_fix_2)) * 5

        else:
            loss_kd = 0

        feat_12 = torch.cat((feat_1, feat_2),dim=1)
        prediction_rec = self.model_c1(feat_12)

        loss_reg_rec = F.mse_loss(prediction, target, reduction='sum') # Apart from the KD loss, there is a supervision (i.e., MSE) loss LSreg on the student model to fulfill the object of the task.


        # ============================================
        #  Train Dis 即训练判别器，用来判断生成器生成的样本是真样本还是假样本。对这些生成的图片，G想要最小化D的输出，而D想要最大化D的输出, 两个网络的目的正好相反，呈现出对抗的姿态。因此这样的训练就叫做 对抗训练(adversarial training), 也叫做GAN。 https://zhuanlan.zhihu.com/p/114838349
        # ============================================
        if epoch % self.args.d_critic == 0:
            loss_D = 0
            ## distinguish by Dl 
            # print(f"[EGSC-KD/src/egsc_kd.py] 正在执行process_batch_adversarial函数，正在训练判别器 feat_1.shape={feat_1.shape}, feat_fix_1.shape={feat_fix_1.shape}, feat_2.shape={feat_2.shape}, feat_fix_2.shape={feat_fix_2.shape}")
            # feat_1.shape=torch.Size([128, 16]), feat_fix_1.shape=torch.Size([128, 16]), feat_2.shape=torch.Size([128, 16]), feat_fix_2.shape=torch.Size([128, 16])
            self.Discriminator.train() # self.Discriminator本质是一个logits判别器，用来判断生成器生成的样本是真样本还是假样本。
            stu_logits = feat_1.detach() # logits.detach()是将logits的梯度设置为False，即不计算logits的梯度。为什么不计算logits的梯度呢？因为logits是生成器生成的样本，而生成器的目的是生成样本，而不是生成样本的梯度。所以不计算logits的梯度。？TODO
            pos_z = self.Discriminator(feat_fix_1) # 将教师模型的输出tea_logits输入到logits判别器中，得到pos_z。这里的pos_z是一个二维张量，表示真样本的logits判别器的输出。
            neg_z = self.Discriminator(stu_logits) # 将学生模型的输出stu_logits输入到logits判别器中，得到neg_z。这里的neg_z是一个二维张量，表示假样本的logits判别器的输出。
            real_z = torch.sigmoid(pos_z[:, -1]) # 这里的real_z是一个一维张量，表示真样本的logits判别器的输出的最后一列，即真样本的logits判别器的输出的最后一个元素。[:, -1] it means you are taking all the rows and only the last column. -1 represents the last column. 
            fake_z = torch.sigmoid(neg_z[:, -1]) # torch.sigmoid()是一个Sigmoid函数，用来将输入的张量中的每个元素转换为0到1之间的数。假样本的logits判别器的输出的最后一个元素。
            ad_loss_1 = self.loss_dis(real_z, torch.ones_like(real_z)) + self.loss_dis(fake_z, torch.zeros_like(fake_z)) # 对抗损失计算逻辑：真样本的logits判别器的输出的最后一个元素与1的交叉熵损失 + 假样本的logits判别器的输出的最后一个元素与0的交叉熵损失。为什么这样算呢？因为真样本的logits判别器的输出的最后一个元素应该趋近于1，而假样本的logits判别器的输出的最后一个元素应该趋近于0。
            
            stu_logits = feat_2.detach() # logits.detach()是将logits的梯度设置为False，即不计算logits的梯度。为什么不计算logits的梯度呢？因为logits是生成器生成的样本，而生成器的目的是生成样本，而不是生成样本的梯度。所以不计算logits的梯度。？TODO
            pos_z = self.Discriminator(feat_fix_2) # 将教师模型的输出tea_logits输入到logits判别器中，得到pos_z。这里的pos_z是一个二维张量，表示真样本的logits判别器的输出。
            neg_z = self.Discriminator(stu_logits) # 将学生模型的输出stu_logits输入到logits判别器中，得到neg_z。这里的neg_z是一个二维张量，表示假样本的logits判别器的输出。
            real_z = torch.sigmoid(pos_z[:, -1]) # 这里的real_z是一个一维张量，表示真样本的logits判别器的输出的最后一列，即真样本的logits判别器的输出的最后一个元素。[:, -1] it means you are taking all the rows and only the last column. -1 represents the last column. 
            fake_z = torch.sigmoid(neg_z[:, -1]) # torch.sigmoid()是一个Sigmoid函数，用来将输入的张量中的每个元素转换为0到1之间的数。假样本的logits判别器的输出的最后一个元素。
            ad_loss_2 = self.loss_dis(real_z, torch.ones_like(real_z)) + self.loss_dis(fake_z, torch.zeros_like(fake_z)) # 对抗损失计算逻辑：真样本的logits判别器的输出的最后一个元素与1的交叉熵损失 + 假样本的logits判别器的输出的最后一个元素与0的交叉熵损失。为什么这样算呢？因为真样本的logits判别器的输出的最后一个元素应该趋近于1，而假样本的logits判别器的输出的最后一个元素应该趋近于0。
            loss_D = 0.5 * (ad_loss_1 + ad_loss_2) # ad_loss全称是adversarial loss，即对抗损失，ds_loss全称是discriminator loss，即判别器损失。loss_D是对抗损失和判别器损失的加权和。
            
            self.opt_D.zero_grad()
            loss_D.backward()  # 反向传播，计算梯度，计算梯度是为了更新模型参数，使得模型参数能够更好的拟合数据，从而提高模型的性能，提高模型的性能是为了提高模型的泛化能力，提高模型的泛化能力是为了提高模型的预测准确率。
            self.opt_D.step()
        # ============================================
        #  Train Stu  训练学生模型，即训练生成器，生成器的目标是欺骗判别器。在训练生成器的时候, 它会调整内部的参数使得它生成的图片越来越真实。也就是它一直在优化使得它产生的数据如图片能够骗过D(判别器), 想要让D认为它生成的图片是真实的。对这些生成的图片，G想要最小化D的输出，而D想要最大化D的输出, 两个网络的目的正好相反，呈现出对抗的姿态。因此这样的训练就叫做 对抗训练(adversarial training), 也叫做GAN。
        # ============================================
        if epoch % self.args.g_critic == 0: # g_critic 全称 generator critic，即生成器的评判者
            loss_G = label_loss
            ## to fool Discriminator_l
            self.Discriminator.eval() # 评估模式，不启用 BatchNormalization 和 Dropout。
            pos_z = self.Discriminator(feat_fix_1) # 将教师模型的输出作为正样本，即真实样本。
            neg_z = self.Discriminator(feat_1) # 将学生模型的输出作为负样本，即生成样本。
            fake_z = torch.sigmoid(neg_z[:, -1]) # 将负样本的最后一列作为生成样本的概率。为什么是最后一列？因为最后一列是生成样本的概率？TODO
            ad_loss = self.loss_dis(fake_z, torch.ones_like(fake_z)) # 计算生成样本(学生模型)的概率与真实样本的概率的BCEloss。通过和真实样本的概率比较，来判断生成样本的概率是不是越来越接近真实样本的概率。查看判别器这里的代码会发现是把生成器和zeros比较。又因为G想要最小化D的输出，而D想要最大化D的输出, 两个网络的目的正好相反，呈现出对抗的姿态。因此对于生成器来说loss是其和1之间的差距，通过减少loss可以让其越来越和1(真实样本)接近，也就是越来越能骗过判别器。而对于判别器来说loss是其和0之间的差距，通过减少loss可以让其越来越和0(生成样本)接近，也就是越来越能区分生成样本和真实样本。判别器的目的是尽可能区分出真实样本和假样本，而生成器的目的是尽可能地欺骗判别器。
            # ds_loss = self.loss_fcn(neg_z[:, :-1][train_mask], labels[train_mask])  # right one # 计算生成样本(学生模型)和真实样本的交叉熵损失。通过和真实样本比较，来判断生成样本是不是越来越接近真实样本。
            l1_loss = torch.norm(feat_1 - feat_fix_1, p=1) * 1. / len(feat_fix_1) # 计算生成样本与真实样本的L1范数。L1范数是指向量各个元素绝对值之和。通过和真实样本比较，来判断生成样本是不是越来越接近真实样本。
            loss_G_1 = loss_G + ad_loss + l1_loss # ad_loss全称adversarial loss，即对抗损失。ds_loss全称discriminator loss，即判别器损失。l1_loss全称L1 loss，即L1损失。

            self.Discriminator.eval() # 评估模式，不启用 BatchNormalization 和 Dropout。
            pos_z = self.Discriminator(feat_fix_2) # 将教师模型的输出作为正样本，即真实样本。
            neg_z = self.Discriminator(feat_2) # 将学生模型的输出作为负样本，即生成样本。
            fake_z = torch.sigmoid(neg_z[:, -1]) # 将负样本的最后一列作为生成样本的概率。为什么是最后一列？因为最后一列是生成样本的概率？TODO
            ad_loss = self.loss_dis(fake_z, torch.ones_like(fake_z)) # 计算生成样本(学生模型)的概率与真实样本的概率的BCEloss。通过和真实样本的概率比较，来判断生成样本的概率是不是越来越接近真实样本的概率。查看判别器这里的代码会发现是把生成器和zeros比较。又因为G想要最小化D的输出，而D想要最大化D的输出, 两个网络的目的正好相反，呈现出对抗的姿态。因此对于生成器来说loss是其和1之间的差距，通过减少loss可以让其越来越和1(真实样本)接近，也就是越来越能骗过判别器。而对于判别器来说loss是其和0之间的差距，通过减少loss可以让其越来越和0(生成样本)接近，也就是越来越能区分生成样本和真实样本。判别器的目的是尽可能区分出真实样本和假样本，而生成器的目的是尽可能地欺骗判别器。
            # ds_loss = self.loss_fcn(neg_z[:, :-1][train_mask], labels[train_mask])  # right one # 计算生成样本(学生模型)和真实样本的交叉熵损失。通过和真实样本比较，来判断生成样本是不是越来越接近真实样本。
            l1_loss = torch.norm(feat_2 - feat_fix_2, p=1) * 1. / len(feat_fix_2) # 计算生成样本与真实样本的L1范数。L1范数是指向量各个元素绝对值之和。通过和真实样本比较，来判断生成样本是不是越来越接近真实样本。
            loss_G_2 = loss_G + ad_loss + l1_loss # ad_loss全称adversarial loss，即对抗损失。ds_loss全称discriminator loss，即判别器损失。l1_loss全称L1 loss，即L1损失。
            loss_G = 0.5 * (loss_G_1 + loss_G_2) # 两个生成器的损失相加。

            # loss_G.backward() # 将loss_G反向传播，更新生成器的参数，使得生成器的损失函数loss_G最小化，即生成器生成的样本越来越接近真实样本。

        # loss = loss_reg + loss_kd + loss_reg_rec # loss_reg是loss_regression loss_kd是loss_knowledge_distillation
        loss = loss_reg + loss_kd + loss_reg_rec + loss_G # loss_reg是loss_regression loss_kd是loss_knowledge_distillation
        loss.backward()
        self.optimizer_c1.step()
        self.optimizer_c.step()
        self.optimizer_f.step()
        self.optimizer_g.step()
        return loss_reg.item(), loss_kd.item()


    def fit(self):
        """
        Training a model.
        """
        print("\nModel training.\n")

        self.optimizer_g = torch.optim.Adam(self.model_g.parameters(), \
            lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.optimizer_f = torch.optim.Adam(self.model_f.parameters(), \
            lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.optimizer_c = torch.optim.Adam(self.model_c.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.optimizer_c1 = torch.optim.Adam(self.model_c1.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.opt_D = torch.optim.Adam(self.Discriminator.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        
        
        self.model_g.train() # model_g 全称是 model_gnn
        self.model_f.train() # model_f 全称是 model_fusion
        self.model_c.train() # model_c 全称是 model_classifier
        self.model_c1.train() # model_c1 全称是 model_classifier1
        
        self.loss_dis = torch.nn.BCELoss()
        self.loss_fcn = torch.nn.CrossEntropyLoss()
        
        epochs = trange(self.args.epochs, leave=True, desc = "Epoch")
        loss_list = []
        loss_list_test = []
        loss_list_kd = []
        loss_list_test_kd = []
        for epoch in epochs:
            
            if self.args.plot:
                if epoch % 10 == 0:
                    self.model_g.train(False)
                    self.model_f.train(False)
                    self.model_c.train(False)
                    cnt_test = 20
                    cnt_train = 100
                    t = tqdm(total=cnt_test*cnt_train, position=2, leave=False, desc = "Validation")
                    scores = torch.empty((cnt_test, cnt_train))
                    
                    for i, g in enumerate(self.testing_graphs[:cnt_test].shuffle()):
                        source_batch = Batch.from_data_list([g]*cnt_train)
                        target_batch = Batch.from_data_list(self.training_graphs[:cnt_train].shuffle())
                        data = self.transform((source_batch, target_batch))
                        target = data["target"]

                        edge_index_1 = data["g1"].edge_index
                        edge_index_2 = data["g2"].edge_index
                        features_1 = data["g1"].x
                        features_2 = data["g2"].x
                        batch_1 = data["g1"].batch if hasattr(data["g1"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes)
                        batch_2 = data["g2"].batch if hasattr(data["g2"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes)

                        pooled_features_1_all = self.model_g(edge_index_1, features_1, batch_1)
                        pooled_features_2_all = self.model_g(edge_index_2, features_2, batch_2)

                        feat_joint = self.model_f(pooled_features_1_all, pooled_features_2_all)
                        prediction = self.model_c(feat_joint)
                        
                        """ print(f'[EGSC-KD/src/egsc_kd.py] 正在执行fit函数 模型输入数据 edge_index_1.shape: {edge_index_1.shape} features_1.shape: {features_1.shape} batch_1.shape: {batch_1.shape}')
                        print(f'[EGSC-KD/src/egsc_kd.py] 正在执行fit函数 经过model_g处理后，输出的pooled_features_1_all.shape: {pooled_features_1_all.shape} pooled_features_2_all.shape: {pooled_features_2_all.shape}')
                        print(f'[EGSC-KD/src/egsc_kd.py] 正在执行fit函数 经过model_f处理后，输出的feat_joint.shape: {feat_joint.shape}')
                        print(f'[EGSC-KD/src/egsc_kd.py] 正在执行fit函数 经过model_c处理后，最终返回的预测结果prediction.shape: {prediction.shape}，目标target.shape: {target.shape}')
                        AIDS700nef 数据集打印结果
                        模型输入数据 edge_index_1.shape: torch.Size([2, 1600]) features_1.shape: torch.Size([800, 29]) batch_1.shape: torch.Size([800])
                        经过model_g处理后，输出的pooled_features_1_all.shape: torch.Size([100, 112]) pooled_features_2_all.shape: torch.Size([100, 112])
                        经过model_f处理后，输出的feat_joint.shape: torch.Size([100, 16])
                        经过model_c处理后，最终返回的预测结果prediction.shape: torch.Size([100])，目标target.shape: torch.Size([100]) """
                        
                        scores[i] = F.mse_loss(prediction, target, reduction='none').detach()
                        t.update(cnt_train)
                    
                    t.close()
                    loss_list_test.append(scores.mean().item())
                    self.model_g.train(True)
                    self.model_f.train(True)
                    self.model_c.train(True)
            
            batches = self.create_batches()
            main_index = 0
            loss_sum = 0
            loss_sum_kd = 0
            for index, batch_pair in tqdm(enumerate(batches), total=len(batches), desc = "Batches"):
                if self.args.use_adversarial == 1:
                    print('[EGSC-KD/src/egsc_kd.py] 准备执行 process_batch_adversarial 函数')
                    loss_score, loss_score_kd = self.process_batch_adversarial(batch_pair, epoch)
                else:
                    print('[EGSC-KD/src/egsc_kd.py] 准备执行 process_batch 函数')
                    loss_score, loss_score_kd = self.process_batch(batch_pair)
                main_index = main_index + batch_pair[0].num_graphs
                loss_sum = loss_sum + loss_score
                loss_sum_kd = loss_sum_kd + loss_score_kd
            loss = loss_sum / main_index
            loss_kd = loss_sum_kd / main_index
            epochs.set_description("Epoch (Loss=%g)" % round(loss,5))
            loss_list.append(loss)
            loss_list_kd.append(loss_kd)
            
        if self.args.plot:
            filename_meta = 'figs/' + self.args.dataset
            filename_meta += '_' + self.args.gnn_operator 
            if self.args.diffpool:
                filename_meta += '_diffpool'
            if self.args.histogram:
                filename_meta += '_hist'
            filename_meta = filename_meta + str(self.args.epochs) + self.args.mode

            plt.plot(loss_list, label="Train")
            plt.plot([*range(0, self.args.epochs, 10)], loss_list_test, label="Validation")

            plt.legend()
            filename = filename_meta + '_mse.pdf'
            plt.savefig(filename)
            plt.close()
            plt.plot([*range(0, self.args.epochs, 1)], loss_list_kd, label="Train KD")

            plt.legend()
            filename = filename_meta + '_kd.pdf'
            plt.savefig(filename)
            plt.close()

            loss_list_kd_array=np.array(loss_list_kd)
            np.save(filename_meta + '_kd.npy',loss_list_kd_array)
            loss_list_array=np.array(loss_list)
            np.save(filename_meta + '_mse_train.npy',loss_list_array)
            loss_list_test_array=np.array(loss_list_test)
            np.save(filename_meta + '_mse_test.npy',loss_list_test_array)

    def score(self):
        """
        Scoring.
        """
        print("\n\nModel evaluation.\n")
        self.model_g.eval()
        self.model_f.eval()
        self.model_c.eval()
        
        scores = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        ground_truth = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        ground_truth_ged = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        prediction_mat = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        
        rho_list = []
        tau_list = []
        prec_at_10_list = [] 
        prec_at_20_list = []
        
        t = tqdm(total=len(self.testing_graphs)*len(self.training_graphs))

        for i, g in enumerate(self.testing_graphs):
            source_batch = Batch.from_data_list([g]*len(self.training_graphs))
            target_batch = Batch.from_data_list(self.training_graphs)
            
            data = self.transform((source_batch, target_batch))
            target = data["target"]
            ground_truth[i] = target
            target_ged = data["target_ged"]
            ground_truth_ged[i] = target_ged

            edge_index_1 = data["g1"].edge_index
            edge_index_2 = data["g2"].edge_index
            features_1 = data["g1"].x
            features_2 = data["g2"].x
            batch_1 = data["g1"].batch if hasattr(data["g1"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes)
            batch_2 = data["g2"].batch if hasattr(data["g2"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes)

            pooled_features_1_all = self.model_g(edge_index_1, features_1, batch_1)
            pooled_features_2_all = self.model_g(edge_index_2, features_2, batch_2)

            feat_joint = self.model_f(pooled_features_1_all, pooled_features_2_all)

            prediction = self.model_c(feat_joint)

            prediction_mat[i] = prediction.detach().numpy()

            scores[i] = F.mse_loss(prediction, target, reduction='none').detach().numpy()

            rho_list.append(calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]))
            tau_list.append(calculate_ranking_correlation(kendalltau, prediction_mat[i], ground_truth[i]))
            prec_at_10_list.append(calculate_prec_at_k(10, prediction_mat[i], ground_truth[i], ground_truth_ged[i]))
            prec_at_20_list.append(calculate_prec_at_k(20, prediction_mat[i], ground_truth[i], ground_truth_ged[i]))

            t.update(len(self.training_graphs))

        self.rho = np.mean(rho_list).item()
        self.tau = np.mean(tau_list).item()
        self.prec_at_10 = np.mean(prec_at_10_list).item()
        self.prec_at_20 = np.mean(prec_at_20_list).item()
        self.model_error = np.mean(scores).item()
        self.print_evaluation()

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        print("\nmse(10^-3): " + str(round(self.model_error*1000, 5)) + ".")
        print("Spearman's rho: " + str(round(self.rho, 5)) + ".")
        print("Kendall's tau: " + str(round(self.tau, 5)) + ".")
        print("p@10: " + str(round(self.prec_at_10, 5)) + ".")
        print("p@20: " + str(round(self.prec_at_20, 5)) + ".")
