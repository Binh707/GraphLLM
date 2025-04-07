import numpy as np
import torch
import random
from torch_geometric.data import Data
import pandas as pd
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_edge_index
import json
from torch_geometric.utils import degree



def load_cora():
    cora_data = torch.load('datasets/cora/cora.pt')
    texts = [
        f"[sep] {text.split(':', 1)[0].strip()} [sep] {text.split(':', 1)[1].strip()}"
        if ':' in text else text
        for text in cora_data.raw_texts
    ]
    #----------------- CORA
    data_name='cora'
    dataset = Planetoid('dataset', data_name,
                            transform=T.NormalizeFeatures())
    data = dataset[0]
    
    data = Data(
        n_id=torch.arange(data.x.shape[0]),
        x=dataset.x,  # ویژگی‌های گره‌ها
        edge_index=data.edge_index,  # یال‌ها
        y=dataset.y,  # برچسب‌های گره‌ها
        train_idx=data.train_mask,
        valid_idx=data.val_mask,
        test_idx=data.test_mask,
        )
    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_idx = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.valid_idx = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_idx = np.sort(node_id[int(data.num_nodes * 0.8):])

    # #  -- splid 60-20-20
    # node_id = np.arange(data.num_nodes)
    # np.random.shuffle(node_id)

    # data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    # data.val_id = np.sort(
    #     node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    # data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

    # data.train_mask = torch.tensor(
    #     [x in data.train_id for x in range(data.num_nodes)])
    # data.val_mask = torch.tensor(
    #     [x in data.val_id for x in range(data.num_nodes)])
    # data.test_mask = torch.tensor(
    #     [x in data.test_id for x in range(data.num_nodes)])
    print("data=",data)
    return data, texts

def load_arxiv(dataset_name):
    dataset = PygNodePropPredDataset(
        name='ogbn-arxiv')
    data = dataset[0]
    # print("data=",data)
    idx_splits = dataset.get_idx_split()

    if dataset_name=='arxiv_sim':
            x = torch.load('datasets/arxiv_sim/x_embs.pt')
            x=x
    else:
            x=data.x
    
    data = Data(
        n_id=torch.arange(data.num_nodes),
        x=x,  # ویژگی‌های گره‌ها
        edge_index= data.edge_index,  # یال‌ها
        y=dataset.y,  # برچسب‌های گره‌ها
        train_idx=idx_splits['train'],
        valid_idx=idx_splits['valid'],
        test_idx=idx_splits['test'],
        )

    nodeidx2paperid = pd.read_csv(
        'datasets/arxiv/nodeidx2paperid.csv.gz', compression='gzip')

    raw_text = pd.read_csv('datasets/arxiv/titleabs.tsv.gz',
                           sep='\t', header=None, names=['paper id', 'title', 'abs'])

    nodeidx2paperid['paper id'] = nodeidx2paperid['paper id'].astype(str)
    raw_text['paper id'] = raw_text['paper id'].astype(str)
    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')

    texts = []
    for ti, ab in zip(df['title'], df['abs']):
        t = '[sep] ' + ti + '[sep]' + ab
        texts.append(t)

    return data, texts

def load_product():
    dataset = PygNodePropPredDataset(
    name='ogbn-products', transform=T.ToSparseTensor())
    data = dataset[0]
    
    idx_splits = dataset.get_idx_split()

    data = Data(
        n_id=torch.arange(data.num_nodes),
        x=data.x,  # ویژگی‌های گره‌ها
        edge_index= data.edge_index,  # یال‌ها
        y=dataset.y,  # برچسب‌های گره‌ها
        train_idx=idx_splits['train'],
        valid_idx=idx_splits['valid'],
        test_idx=idx_splits['test'],
        )
    print("data=",data)
    # data = torch.load('dataset/ogbn_products/ogbn-products_subset.pt')
    text = pd.read_csv('datasets/products/ogbn-products_subset.csv')
    texts = [f'[sep] {ti}. [sep] {cont}'for ti,
            cont in zip(text['title'], text['content'])]
    print("text=",texts[0])
    return data, texts


def load_products_subset():

    data = torch.load('datasets/products/ogbn-products_subset.pt')
    node_desc = pd.read_csv('datasets/products/ogbn-products_subset.csv')

    train_mask, val_mask, test_mask = data.train_mask.squeeze(), data.val_mask.squeeze(), data.test_mask.squeeze()
    
    texts = []
    for i in range(data.num_nodes):
            node_title = (node_desc.iloc[i, 2] if node_desc.iloc[i, 2] is not np.nan else "missing")
            node_content = (node_desc.iloc[i, 3] if node_desc.iloc[i, 3] is not np.nan else "missing")
            text = "[sep] " + node_title + " [sep] " + node_content
            texts.append(text)
    edge_index = data.adj_t.to_symmetric()
    edge_index = to_edge_index(edge_index)[0]
    data = Data(
        n_id=torch.arange(data.num_nodes),
        x=data.x,  # ویژگی‌های گره‌ها
        edge_index= edge_index,  # یال‌ها
        y=data.y,  # برچسب‌های گره‌ها
        train_idx=torch.where(data.train_mask)[0],
        valid_idx=torch.where(data.val_mask)[0],
        test_idx=torch.where(data.test_mask)[0],
        )
    return data, texts






def load_pubmed():
    
    dataset = Planetoid('dataset', 'PubMed', transform=T.NormalizeFeatures())
    data = dataset[0]
    print("data",data)

    data = Data(
        n_id=torch.arange(data.x.shape[0]),
        x=dataset.x,  # ویژگی‌های گره‌ها
        edge_index=data.edge_index,  # یال‌ها
        y=dataset.y,  # برچسب‌های گره‌ها
        # train_idx=torch.where(data.train_mask)[0],
        # valid_idx=torch.where(data.val_mask)[0],
        # test_idx=torch.where(data.test_mask)[0],
        )
    print("data",data)
    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_idx = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.valid_idx = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_idx = np.sort(node_id[int(data.num_nodes * 0.8):])

    f = open('datasets/PubMed/pubmed.json')
    pubmed = json.load(f)
    df_pubmed = pd.DataFrame.from_dict(pubmed)

    AB = df_pubmed['AB'].fillna("")
    TI = df_pubmed['TI'].fillna("")
    texts = []
    for ti, ab in zip(TI, AB):
        t = '[sep] ' + ti + '[sep] ' + ab
        texts.append(t)
    print("text",texts[0])



    return data, texts


def load_arxiv_2023():
    data = torch.load('datasets/arxiv_2023/graph.pt')
    print("data",data)
    # split data
    data.num_nodes = len(data.y)
    num_nodes = data.num_nodes
    node_id = np.arange(num_nodes)
    np.random.shuffle(node_id)
    data = Data(
        n_id=torch.arange(data.x.shape[0]),
        x=data.x,  # ویژگی‌های گره‌ها
        edge_index=data.edge_index,  # یال‌ها
        y=data.y,  # برچسب‌های گره‌ها
        # train_idx=data.train_id,
        # valid_idx=data.val_id,
        # test_idx=data.test_id,
        )
    print("data",data)
    data.train_idx = np.sort(node_id[:int(num_nodes * 0.6)])
    data.valid_idx = np.sort(
        node_id[int(num_nodes * 0.6):int(num_nodes * 0.8)])
    data.test_idx = np.sort(node_id[int(num_nodes * 0.8):])



        # محاسبه میانگین درجه نودها برای گراف بدون جهت
    edge_index = data.edge_index
    # degrees = degree(edge_index.view(-1), num_nodes=num_nodes)
    degrees = degree(edge_index.reshape(-1), num_nodes=num_nodes)

    avg_degree = degrees.mean().item()
    print(f"Average node degree (undirected): {avg_degree:.4f}")

    # data.train_mask = torch.tensor(
    #     [x in data.train_id for x in range(num_nodes)])
    # data.val_mask = torch.tensor(
    #     [x in data.val_id for x in range(num_nodes)])
    # data.test_mask = torch.tensor(
    #     [x in data.test_id for x in range(num_nodes)])

    df = pd.read_csv('datasets/arxiv_2023/paper_info.csv')
    texts = []
    for ti, ab in zip(df['title'], df['abstract']):
        texts.append(f'[sep] {ti}[sep] {ab}')
    return data, texts
