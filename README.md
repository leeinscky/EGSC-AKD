# Efficient Graph Similarity Computation via Adversarial Knowledge Distillation - (EGSC-AKD)

This repo is developed based on [Efficient_Graph_Similarity_Computation](https://github.com/canqin001/Efficient_Graph_Similarity_Computation)

![EGSC](Figs/our-setting.png)

## Training and Evaluation
[1. Train & Test with Teacher Model](https://github.com/leeinscky/EGSC-AKD/blob/main/EGSC-T/README.md)

[2. Train & Test with Student Model](https://github.com/leeinscky/EGSC-AKD/blob/main/EGSC-KD/README.md)

## Introduction
<div>
    <br>
Graph Similarity Computing (GSC) has always been an important task in graph computing. Traditional methods to solve GSC tasks have a high computational complexity, so they are difficult to deploy in real-world industrial scenarios which often have graphs with large sizes and require a strict inference time. Based on an already proposed GNN-based model (EGSC-Sr'f) for GSC tasks, this paper is the first attempt to introduce Adversarial Knowledge Distillation (AKD) into the student model (EGSC-S). Experimental results show that our proposed method outperforms existing student models in most scenarios and has a matching inference time. Therefore, this work provides a good direction for the compression and deployment of GSC computational models in industrial scenarios.
    <br>
</div>

## Dataset
We have used the standard dataloader, i.e., ‘GEDDataset’, directly provided in the [PyG](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/ged_dataset.html#GEDDataset).

```  AIDS700nef:  ``` https://drive.google.com/uc?export=download&id=10czBPJDEzEDI2tq7Z7mkBjLhj55F-a2z

```  LINUX:  ``` https://drive.google.com/uc?export=download&id=1nw0RRVgyLpit4V4XFQyDy0pI6wUEXSOI

```  ALKANE:  ``` https://drive.google.com/uc?export=download&id=1-LmxaWW3KulLh00YqscVEflbqr0g4cXt

```  IMDBMulti:  ``` https://drive.google.com/uc?export=download&id=12QxZ7EhYA7pJiF4cO-HuE8szhSOWcfST


<p align="justify">
The code takes pairs of graphs for training from an input folder where each pair of graph is stored as a JSON. Pairs of graphs used for testing are also stored as JSON files. Every node id and node label has to be indexed from 0. Keys of dictionaries are stored strings in order to make JSON serialization possible.</p>

Every JSON file has the following key-value structure:

```javascript
{"graph_1": [[0, 1], [1, 2], [2, 3], [3, 4]],
 "graph_2":  [[0, 1], [1, 2], [1, 3], [3, 4], [2, 4]],
 "labels_1": [2, 2, 2, 2],
 "labels_2": [2, 3, 2, 2, 2],
 "ged": 1}
```
<p align="justify">
The **graph_1** and **graph_2** keys have edge list values which descibe the connectivity structure. Similarly, the **labels_1**  and **labels_2** keys have labels for each node which are stored as list - positions in the list correspond to node identifiers. The **ged** key has an integer value which is the raw graph edit distance for the pair of graphs.</p>

## Requirements
The codebase is implemented in Python 3.9.0. package versions used for development are just below.
```
certifi==2022.12.7
charset-normalizer==2.1.1
click==8.1.3
cycler==0.11.0
decorator==5.1.1
dgl==0.9.1
docker-pycreds==0.4.0
gitdb==4.0.10
GitPython==3.1.30
idna==3.4
Jinja2==3.1.2
joblib==1.2.0
kiwisolver==1.4.4
MarkupSafe==2.1.1
matplotlib==3.3.4
mkl-fft==1.3.1
mkl-random==1.2.2
mkl-service==2.4.0
networkx==2.4
pandas==1.5.2
pathtools==0.1.2
promise==2.3
protobuf==4.21.12
psutil==5.9.4
pyparsing==3.0.9
python-dateutil==2.8.2
pytz==2022.7
PyYAML==6.0
requests==2.28.1
scikit-learn==0.23.2
scipy==1.9.3
sentry-sdk==1.12.1
setproctitle==1.3.2
shortuuid==1.0.11
smmap==5.0.0
texttable==1.6.3
thop==0.1.1.post2209072238
threadpoolctl==3.1.0
torch==1.10.1
torch-geometric==2.2.0
torchaudio==0.10.1
torchvision==0.11.2
tqdm==4.60.0
urllib3==1.26.13
wandb==0.13.7
```

## File Structure
```
.
├── README.md
├── LICENSE                            
├── EGSC-T
│   ├── src
│   │    ├── egsc.py 
│   │    ├── layers.py
│   │    ├── main.py
│   │    ├── model.py
│   │    ├── parser.py        
│   │    └── utils.py                             
│   ├── README.md                      
│   └── train.sh
├── EGSC-KD
│   ├── src
│   │    ├── egsc_kd.py 
│   │    ├── egsc_nonkd.py 
│   │    ├── layers.py
│   │    ├── main_kd.py
│   │    ├── main_nonkd.py
│   │    ├── model_kd.py
│   │    ├── parser.py    
│   │    ├── trans_modules.py    
│   │    └── utils.py                             
│   ├── README.md  
│   ├── train_kd.md                     
│   └── train_nonkd.sh 
├── Checkpoints
│   ├── G_EarlyFusion_Disentangle_LINUX_gin_checkpoint.pth
│   ├── G_EarlyFusion_Disentangle_IMDBMulti_gin_checkpoint.pth
│   ├── G_EarlyFusion_Disentangle_ALKANE_gin_checkpoint.pth
│   └── G_EarlyFusion_Disentangle_AIDS700nef_gin_checkpoint.pth                         
└── GSC_datasets
    ├── AIDS700nef
    ├── ALKANE
    ├── IMDBMulti
    └── LINUX
```

## Acknowledgement
We would like to thank the [Efficient_Graph_Similarity_Computation
](https://github.com/canqin001/Efficient_Graph_Similarity_Computation)which we used for this implementation.

## Hint
On some datasets, the results are not quite stable. We suggest to run multiple times to report the avarage one.
