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

from model import EGSCT_generator, EGSCT_classifier

import pdb

class EGSCTrainer(object):
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.process_dataset()
        self.setup_model()
        self.best_rho = 0
        self.best_tau = 0
        self.best_prec_at_10 = 0
        self.best_prec_at_20 = 0
        self.best_model_error = float('inf')


    def setup_model(self):
        """
        Creating a EGSC.
        """
        self.model_g = EGSCT_generator(self.args, self.number_of_labels)
        self.model_c = EGSCT_classifier(self.args, self.number_of_labels)
        # print(f'[EGSC-T/src/egsc.py] 正在执行setup_model函数 self.model_g={self.model_g}, self.model_c={self.model_c}')
        """ [EGSC-T/src/egsc.py] 正在执行setup_model函数 self.model_g=EGSCT_generator(
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
        (attention_level3): AttentionModule(
            (fc): Sequential(
            (0): Linear(in_features=16, out_features=4, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=4, out_features=16, bias=True)
            (3): Tanh()
            )
            (fc1): Linear(in_features=16, out_features=16, bias=True)
        )
        (attention_level2): AttentionModule(
            (fc): Sequential(
            (0): Linear(in_features=32, out_features=8, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=8, out_features=32, bias=True)
            (3): Tanh()
            )
            (fc1): Linear(in_features=32, out_features=32, bias=True)
        )
        (attention_level1): AttentionModule(
            (fc): Sequential(
            (0): Linear(in_features=64, out_features=16, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=16, out_features=64, bias=True)
            (3): Tanh()
            )
            (fc1): Linear(in_features=64, out_features=64, bias=True)
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
        ), self.model_c=EGSCT_classifier(
        (scoring_layer): Linear(in_features=16, out_features=1, bias=True)
        ) """

    def save_model(self):
        """
        Saving a EGSC.
        """
        PATH_g = './model_saved/EGSC_g_EarlyFusion_' +str(self.args.dataset)+"_"+str(self.args.gnn_operator)+"_"+ str(round(self.model_error*1000, 5))+"_" \
        + str(self.args.epochs)+"_"+str(self.args.batch_size)+"_"+str(self.args.learning_rate) +'_checkpoint.pth'
        torch.save(self.model_g.state_dict(), PATH_g)

        PATH_c = './model_saved/EGSC_c_EarlyFusion_' +str(self.args.dataset)+"_"+str(self.args.gnn_operator)+"_"+ str(round(self.model_error*1000, 5))+"_" \
        + str(self.args.epochs)+"_"+str(self.args.batch_size)+"_"+str(self.args.learning_rate) +'_checkpoint.pth'
        torch.save(self.model_c.state_dict(), PATH_c)
        print('Model Saved')

    def load_model(self):
        """
        Loading a EGSC.
        """
        PATH_g = './model_saved/EGSC_g_EarlyFusion_' +str(self.args.dataset)+"_"+str(self.args.gnn_operator)+"_"+ str(round(self.model_error*1000, 5))+"_" \
        + str(self.args.epochs)+"_"+str(self.args.batch_size)+"_"+str(self.args.learning_rate) +'_checkpoint.pth'
        self.model_g.load_state_dict(torch.load(PATH_g))

        PATH_c = './model_saved/EGSC_c_EarlyFusion_' +str(self.args.dataset)+"_"+str(self.args.gnn_operator)+"_"+ str(round(self.model_error*1000, 5))+"_" \
        + str(self.args.epochs)+"_"+str(self.args.batch_size)+"_"+str(self.args.learning_rate) +'_checkpoint.pth'
        self.model_c.load_state_dict(torch.load(PATH_c))
        print('Model Loaded')
        
    def process_dataset(self):
        """
        Downloading and processing dataset.
        """
        print("\nPreparing dataset.\n")

        self.args.data_dir = '../GSC_datasets'

        self.training_graphs = GEDDataset(self.args.data_dir+'/{}'.format(self.args.dataset), self.args.dataset, train=True) 
        self.testing_graphs = GEDDataset(self.args.data_dir+'/{}'.format(self.args.dataset), self.args.dataset, train=False) 
        if self.args.dataset=="ALKANE":
            self.testing_graphs = GEDDataset(self.args.data_dir+'/{}'.format(self.args.dataset), self.args.dataset, train=True) 
        
        # self.testing_graphs.norm_ged
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
        """
        Creating batches from the training graph list.
        :return batches: Zipped loaders as list.
        """
        if self.args.synth:
            synth_data_ind = random.sample(range(len(self.synth_data_1)), 100)
        
        source_loader = DataLoader(self.training_graphs.shuffle() + 
            ([self.synth_data_1[i] for i in synth_data_ind] if self.args.synth else []), batch_size=self.args.batch_size)
        target_loader = DataLoader(self.training_graphs.shuffle() + 
            ([self.synth_data_2[i] for i in synth_data_ind] if self.args.synth else []), batch_size=self.args.batch_size)
        
        return list(zip(source_loader, target_loader))

    def transform(self, data):
        """
        Getting ged for graph pair and grouping with data into dictionary.
        :param data: Graph pair.
        :return new_data: Dictionary with data.
        """
        new_data = dict()

        new_data["g1"] = data[0]
        new_data["g2"] = data[1]

        normalized_ged = self.nged_matrix[data[0]["i"].reshape(-1).tolist(),data[1]["i"].reshape(-1).tolist()].tolist()
        
        new_data["target"] = torch.from_numpy(np.exp([(-el) for el in normalized_ged])).view(-1).float()
        ged = self.ged_matrix[data[0]["i"].reshape(-1).tolist(),data[1]["i"].reshape(-1).tolist()].tolist()

        new_data["target_ged"] = torch.from_numpy(np.array([(el) for el in ged])).view(-1).float()

        return new_data

    def process_batch(self, data):
        """
        Forward pass with a data.
        :param data: Data that is essentially pair of batches, for source and target graphs.
        :return loss: Loss on the data. 
        """
        self.optimizer.zero_grad()
        
        data = self.transform(data)
        target = data["target"]

        
        prediction = self.model_c(self.model_g(data))
        loss = F.mse_loss(prediction, target, reduction='sum') #* 0.5
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def fit(self):
        """
        Training a model.
        """
        print("\nModel training.\n")
        self.optimizer = torch.optim.Adam([{'params': self.model_g.parameters()}, {'params': self.model_c.parameters()}],\
         lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.model_g.train()
        self.model_c.train()

        
        epochs = trange(self.args.epochs, leave=True, desc = "Epoch")
        loss_list = []
        loss_list_test = []
        for epoch in epochs:
            if self.args.plot:
                if epoch % 10 == 0: # 每10个epoch进行一次记录（原始逻辑）
                    self.model_g.train(False) # model_g全称是model_graph，即图神经网络 False代表不训练，即测试模式，True代表训练模式。这是为了防止测试时出现梯度爆炸
                    self.model_c.train(False) # model_c全称是model_classifier，即分类器
                    cnt_test = 20 # cnt_test全称是count_test，即测试集的数量
                    cnt_train = 100 
                    t = tqdm(total=cnt_test*cnt_train, position=2, leave=False, desc = "Validation")
                    scores = torch.empty((cnt_test, cnt_train))
                    
                    for i, g in enumerate(self.testing_graphs[:cnt_test].shuffle()): # i是索引，g是图 self.testing_graphs[:cnt_test]表示取前cnt_test个图，shuffle()表示打乱顺序了
                        source_batch = Batch.from_data_list([g]*cnt_train) # 生成一个batch，里面有cnt_train个图, source_batch是源图, target_batch是目标图
                        target_batch = Batch.from_data_list(self.training_graphs[:cnt_train].shuffle()) # 生成一个batch，里面有cnt_train个图
                        data = self.transform((source_batch, target_batch)) # data是一个字典，里面有g1, g2, target, target_ged, transform函数是将图对转换为字典
                        target = data["target"]
                        prediction = self.model_c(self.model_g(data))
                        
                        # tqdm.write(f'[EGSC-T/src/egsc.py] 正在执行fit函数，第{epoch}个epoch，第{i}个图，输入数据为{data}, 预测值为{prediction}，真实值为{target}')
                        """ dataset AIDS700nef 打印信息
                        输入数据为{
                                'g1': DataBatch(edge_index=[2, 1600], i=[100], x=[800, 29], num_nodes=800, batch=[800], ptr=[101]), 
                                'g2': DataBatch(edge_index=[2, 1620], i=[100], x=[835, 29], num_nodes=835, batch=[835], ptr=[101]), 
                                'target': tensor([0.4111, 0.2865, ...]),  # torch.Size([128])
                                'target_ged': tensor([ 8., 10.,  6., ...])},  # torch.Size([128])
                        预测值为tensor([0.4595, 0.4602, 0.4595, ...], grad_fn=<ViewBackward0>)， # torch.Size([128])
                        真实值为tensor([0.4111, 0.2865, 0.4493, ...]) """ 
                        
                        scores[i] = F.mse_loss(prediction, target, reduction='none').detach() # mse_loss计算过程具体为：(x - y)^2，这里的reduction='none'表示不进行求和，而是直接返回每个元素的平方
                        t.update(cnt_train) # 更新进度条
                    
                    t.close() # 关闭进度条
                    loss_list_test.append(scores.mean().item()) # 计算测试集的loss
                    self.model_g.train(True) # 训练模式，即测试完毕后，将模型切换为训练模式，以便进行下一轮的训练
                    self.model_c.train(True)
            
            batches = self.create_batches() # 生成batch
            main_index = 0 
            loss_sum = 0 
            for index, batch_pair in tqdm(enumerate(batches), total=len(batches), desc = "Batches"): # 这个循环目的是计算一个batch的loss
                loss_score = self.process_batch(batch_pair) # 计算一个batch的loss
                main_index = main_index + batch_pair[0].num_graphs # batch_pair[0]是源图，batch_pair[1]是目标图
                loss_sum = loss_sum + loss_score # loss_sum是所有batch的loss之和
            loss = loss_sum / main_index 
            epochs.set_description("Epoch (Loss=%g)" % round(loss,5)) # 设置进度条的描述
            loss_list.append(loss) # 将每一轮的loss加入到loss_list中
            
        if self.args.plot:
            plt.plot(loss_list, label="Train")
            plt.plot([*range(0, self.args.epochs, 10)], loss_list_test, label="Validation")
            plt.ylim([0, 0.01])
            plt.legend()
            filename = self.args.dataset
            filename += '_' + self.args.gnn_operator 
            filename = filename + str(self.args.epochs) + '.pdf'
            # print(f'[EGSC-T/src/egsc.py] 正在执行fit函数，正在保存loss图像，文件名为{filename}')
            plt.savefig(filename)

    def score(self):
        """
        Scoring.
        """
        print("\n\nModel evaluation.\n")

        self.model_g.eval() # 将模型切换为评估模式, 即测试模式, 模型不会更新参数
        self.model_c.eval()
        
        scores = np.empty((len(self.testing_graphs), len(self.training_graphs))) # 生成一个空的矩阵，用于存放测试集和训练集的loss
        ground_truth = np.empty((len(self.testing_graphs), len(self.training_graphs))) # 生成一个空的矩阵，用于存放测试集和训练集的真实值
        ground_truth_ged = np.empty((len(self.testing_graphs), len(self.training_graphs))) # 生成一个空的矩阵，用于存放测试集和训练集的ged ged全称是graph edit distance
        prediction_mat = np.empty((len(self.testing_graphs), len(self.training_graphs))) # 生成一个空的矩阵，用于存放测试集和训练集的预测值
        
        rho_list = [] # 生成一个空的列表，用于存放测试集和训练集的rho值 rho全称是spearman相关系数
        tau_list = [] # 生成一个空的列表，用于存放测试集和训练集的tau值 tau全称是kendall相关系数
        prec_at_10_list = [] # 生成一个空的列表，用于存放测试集和训练集的prec_at_10值 prec_at_10全称是top10的准确率
        prec_at_20_list = [] # 生成一个空的列表，用于存放测试集和训练集的prec_at_20值 prec_at_20全称是top20的准确率
        
        t = tqdm(total=len(self.testing_graphs)*len(self.training_graphs))

        for i, g in enumerate(self.testing_graphs): 
            source_batch = Batch.from_data_list([g]*len(self.training_graphs)) # 生成一个batch，[g]*len(self.training_graphs)表示将g重复len(self.training_graphs)次, 生成一个长度为len(self.training_graphs)的列表, 列表中的每个元素都是g, 这个列表就是source_batch中的图, 也就是说source_batch中包含了len(self.training_graphs)个g, 这些g都是测试集中的图
            target_batch = Batch.from_data_list(self.training_graphs) # 生成一个batch，里面包含了训练集中的所有图, 这些图都是训练集中的图
            
            data = self.transform((source_batch, target_batch))
            target = data["target"]
            ground_truth[i] = target
            target_ged = data["target_ged"]
            ground_truth_ged[i] = target_ged

            prediction = self.model_c(self.model_g(data)) 
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
