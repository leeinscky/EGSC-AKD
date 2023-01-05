import torch
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
from scipy.stats import spearmanr, kendalltau

from layers import DiffPool, ConfusionAttentionModule
from utils import calculate_ranking_correlation, calculate_prec_at_k, gen_pairs

from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv
from torch_geometric.data import DataLoader, Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import OneHotDegree

import pdb
from layers import AttentionModule as AttentionModule
from layers import SETensorNetworkModule as TensorNetworkModule
from layers import SEAttentionModule

class EGSCT_generator(torch.nn.Module):
    def __init__(self, args, number_of_labels):
        super(EGSCT_generator, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.dim_aug_feats = 0
        self.scaler_dim = 1
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        self.feature_count = (self.args.filters_1 + self.args.filters_2 + self.args.filters_3 ) // 2

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        if self.args.gnn_operator == 'gcn':
            self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        elif self.args.gnn_operator == 'gin':
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.number_labels, self.args.filters_1), 
                torch.nn.ReLU(), 
                torch.nn.Linear(self.args.filters_1, self.args.filters_1),
                torch.nn.BatchNorm1d(self.args.filters_1))
            
            nn2 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_1, self.args.filters_2), 
                torch.nn.ReLU(), 
                torch.nn.Linear(self.args.filters_2, self.args.filters_2),
                torch.nn.BatchNorm1d(self.args.filters_2))
            
            nn3 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_2, self.args.filters_3), 
                torch.nn.ReLU(), 
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.BatchNorm1d(self.args.filters_3))
            
            self.convolution_1 = GINConv(nn1, train_eps=True)
            self.convolution_2 = GINConv(nn2, train_eps=True)
            self.convolution_3 = GINConv(nn3, train_eps=True)
        elif self.args.gnn_operator == 'gat':
            self.convolution_1 = GATConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = GATConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GATConv(self.args.filters_2, self.args.filters_3)
        elif self.args.gnn_operator == 'sage':
            self.convolution_1 = SAGEConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = SAGEConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = SAGEConv(self.args.filters_2, self.args.filters_3)

        else:
            raise NotImplementedError('Unknown GNN-Operator.')

        self.attention_level3 = AttentionModule(self.args, self.args.filters_3)

        self.attention_level2 = AttentionModule(self.args, self.args.filters_2 * self.scaler_dim)

        self.attention_level1 = AttentionModule(self.args, self.args.filters_1 * self.scaler_dim)

        self.tensor_network_level3 = TensorNetworkModule(self.args,dim_size=self.args.filters_3 * self.scaler_dim)
        self.tensor_network_level2 = TensorNetworkModule(self.args,dim_size=self.args.filters_2 * self.scaler_dim)
        self.tensor_network_level1 = TensorNetworkModule(self.args,dim_size=self.args.filters_1 * self.scaler_dim)
        self.fully_connected_first = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

        self.score_attention = SEAttentionModule(self.args, self.feature_count)


    def convolutional_pass_level1(self, edge_index, features):
        """
        Making convolutional pass.
        """
        features = self.convolution_1(features, edge_index)
        features = F.relu(features)
        features_1 = F.dropout(features, p=self.args.dropout, training=self.training)
        return features_1

    def convolutional_pass_level2(self, edge_index, features):
        features_2 = self.convolution_2(features, edge_index)
        features_2 = F.relu(features_2)
        features_2 = F.dropout(features_2, p=self.args.dropout, training=self.training)
        return features_2

    def convolutional_pass_level3(self, edge_index, features):
        features_3 = self.convolution_3(features, edge_index)
        features_3 = F.relu(features_3)
        features_3 = F.dropout(features_3, p=self.args.dropout, training=self.training)
        return features_3

    def convolutional_pass_level4(self, edge_index, features):
        features_out = self.convolution_4(features, edge_index)
        return features_out
        
    def forward(self, data):
        # print(f'[EGSC-T/src/model.py] 生成器 EGSCT_generator 正在执行forward函数，输入的data为{data}')
        # print(f"[EGSC-T/src/model.py] 生成器 EGSCT_generator 正在执行forward函数，data[target].shape为{data['target'].shape}, data[target_ged].shape为{data['target_ged'].shape}")
        edge_index_1 = data["g1"].edge_index
        edge_index_2 = data["g2"].edge_index
        features_1 = data["g1"].x
        features_2 = data["g2"].x
        batch_1 = data["g1"].batch if hasattr(data["g1"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes)
        batch_2 = data["g2"].batch if hasattr(data["g2"], 'batch') else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes)
        
        features_level1_1 = self.convolutional_pass_level1(edge_index_1, features_1)
        features_level1_2 = self.convolutional_pass_level1(edge_index_2, features_2)
        # print(f'[EGSC-T/src/model.py] 生成器 EGSCT_generator 正在执行forward函数， features_level1_1.shape为{features_level1_1.shape}, features_level1_2.shape为{features_level1_2.shape}')
        
        pooled_features_level1_1 = self.attention_level1(features_level1_1, batch_1) # 128 * 64
        pooled_features_level1_2 = self.attention_level1(features_level1_2, batch_2) # 128 * 64
        scores_level1 = self.tensor_network_level1(pooled_features_level1_1, pooled_features_level1_2)
        # print(f'[EGSC-T/src/model.py] 生成器 EGSCT_generator 正在执行forward函数，pooled_features_level1_1.shape为{pooled_features_level1_1.shape}, pooled_features_level1_2.shape为{pooled_features_level1_2.shape}, scores_level1.shape为{scores_level1.shape}')

        features_level2_1 = self.convolutional_pass_level2(edge_index_1, features_level1_1)
        features_level2_2 = self.convolutional_pass_level2(edge_index_2, features_level1_2)
        # print(f'[EGSC-T/src/model.py] 生成器 EGSCT_generator 正在执行forward函数，features_level2_1.shape为{features_level2_1.shape}, features_level2_2.shape为{features_level2_2.shape}')

        pooled_features_level2_1 = self.attention_level2(features_level2_1, batch_1) # 128 * 32
        pooled_features_level2_2 = self.attention_level2(features_level2_2, batch_2) # 128 * 32
        scores_level2 = self.tensor_network_level2(pooled_features_level2_1, pooled_features_level2_2)
        # print(f'[EGSC-T/src/model.py] 生成器 EGSCT_generator 正在执行forward函数，pooled_features_level2_1.shape为{pooled_features_level2_1.shape}, pooled_features_level2_2.shape为{pooled_features_level2_2.shape}, scores_level2.shape为{scores_level2.shape}')

        features_level3_1 = self.convolutional_pass_level3(edge_index_1, features_level2_1)
        features_level3_2 = self.convolutional_pass_level3(edge_index_2, features_level2_2)
        pooled_features_level3_1 = self.attention_level3(features_level3_1, batch_1) # 128 * 16
        pooled_features_level3_2 = self.attention_level3(features_level3_2, batch_2) # 128 * 16
        scores_level3 = self.tensor_network_level3(pooled_features_level3_1, pooled_features_level3_2)
        # print(f'[EGSC-T/src/model.py] 生成器 EGSCT_generator 正在执行forward函数，pooled_features_level3_1.shape为{pooled_features_level3_1.shape}, pooled_features_level3_2.shape为{pooled_features_level3_2.shape}, scores_level3.shape为{scores_level3.shape}')
        
        scores = torch.cat((scores_level3, scores_level2, scores_level1), dim=1)
        # print(f'[EGSC-T/src/model.py] 生成器 EGSCT_generator 正在执行forward函数，scores.shape为{scores.shape}')
        
        scores = F.relu(self.fully_connected_first(self.score_attention(scores)*scores + scores))
        # print(f'[EGSC-T/src/model.py] 生成器 EGSCT_generator 正在执行forward函数，最终返回的scores.shape={scores.shape}')

        """ dataset AIDS700nef 打印信息汇总
        输入的data为{
                    'g1': DataBatch(
                        edge_index=[2, 2252], i=[128], x=[1133, 29],
                        num_nodes=1133, batch=[1133], ptr=[129]),
                    'g2': DataBatch(
                        edge_index=[2, 2224], i=[128], x=[1121, 29],
                        num_nodes=1121, batch=[1121], ptr=[129]),
                    'target': tensor([0.4966, 0.4066, 0.531...])  # data[target].shape为torch.Size([128])
                    'target_ged': tensor([ 7.,  9.,  6.,  9., 13.,  9., ...]) # data[target_ged].shape为torch.Size([128])
                }
        features_level1_1.shape: torch.Size([1133, 64])
        features_level1_2.shape: torch.Size([1121, 64])
        pooled_features_level1_1.shape: torch.Size([128, 64])
        pooled_features_level1_2.shape: torch.Size([128, 64])
        scores_level1.shape: torch.Size([128, 32])
        
        features_level2_1.shape: torch.Size([1133, 32])
        features_level2_2.shape: torch.Size([1121, 32])
        pooled_features_level2_1.shape: torch.Size([128, 32])
        pooled_features_level2_2.shape: torch.Size([128, 32])
        scores_level2.shape: torch.Size([128, 16])
        
        pooled_features_level3_1.shape: torch.Size([128, 16])
        pooled_features_level3_2.shape: torch.Size([128, 16])
        scores_level3.shape: torch.Size([128, 8])
        
        scores.shape: torch.Size([128, 56])
        最终返回的scores.shape=torch.Size([128, 16]) """
        return  scores 

class EGSCT_classifier(torch.nn.Module):
    def __init__(self, args, number_of_labels):
        super(EGSCT_classifier, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.dim_aug_feats = 0
        self.scaler_dim = 1
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def forward(self, scores):
        score = torch.sigmoid(self.scoring_layer(scores)).view(-1) # dim of score: 128 * 0
        # print(f'[EGSC-T/src/model.py] 分类器 EGSCT_classifier 正在执行forward函数 输入scores的维度为{scores.shape}, 输出score的维度为{score.shape}')
        #  输入scores的维度为torch.Size([128, 16]), 输出score的维度为torch.Size([128])
        return  score 