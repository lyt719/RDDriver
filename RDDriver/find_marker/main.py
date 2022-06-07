import torch
import warnings
import argparse
import numpy as np
import pandas as pd
import statsmodels.stats.weightstats as sw
from load import load_random_hetero_graph
import os
import mvc_env
import networkx as nx
from load import *
from DQN import DQN
import dgl
import pickle
from model import RGCNDQNModel
import matplotlib.pyplot as plt
from eval import *
import sys
import collections
sys.path.append('../')
warnings.filterwarnings("ignore")


def model_sort():
    env = mvc_env.MVC_env()
    init_graph = load_random_hetero_graph(args['path'], Specified_quantity={'lncrna': 1500, 'mirna': 168, 'mrna': 2519})
    dqn = DQN(args, init_graph.number_of_nodes())
    g, s, num_rna = env.reset(init_graph)


    param_path_strlist=['./save_model/9999_-13.pth','./save_model/9999_-15.pth','./save_model/9999_-22.pth']
    actions_value=[[0]]*g.number_of_nodes()
    for pp in param_path_strlist:
        dqn.eval_net.load_state_dict(torch.load(pp, map_location='cpu'))
        dqn.eval_net=dqn.eval_net.cuda()
        G = dgl.to_homogeneous(init_graph).to_networkx().to_undirected()
        nodes1 = [n for n, v_ in G.degree() if v_ != 0]
        f_mirna = torch.tensor([[-1.0] * args['in_size'] for o in range(g.number_of_nodes('mirna'))]).to(args['device'])
        f_mrna = torch.tensor([[-1.0] * args['in_size'] for o in range(g.number_of_nodes('mrna'))]).to(args['device'])
        f_lncrna = torch.tensor([[-1.0] * args['in_size'] for o in range(g.number_of_nodes('lncrna'))]).to(args['device'])
        h = {'mirna': f_mirna, 'mrna': f_mrna, 'lncrna': f_lncrna}
        actions_value1 = dqn.eval_net.forward(args, g, h).detach().cpu().numpy().tolist()
        actions_value=(np.sum([actions_value, actions_value1], axis=0)).tolist()

    actions_value=(np.sum([actions_value], axis=0)/3).tolist()
    all_values=[]
    for index,value in zip(nodes1,actions_value):
        all_values+=[(index,actions_value[index])]
    sorted_value = sorted(all_values, key=lambda x: (x[1], x[0]), reverse=True)
    action_choosed = [x[0] for x in sorted_value]
    return action_choosed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_size', type=int, default=1)
    parser.add_argument('--out_size', type=int, default=1)
    parser.add_argument('--feat_dim', type=int, default=32)
    parser.add_argument('--num_bases', type=int, default=5)
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--all_relations', type=list,default=['lnc_lnc', 'lnc_mi', 'm_lnc', 'mi_lnc', 'mi_m', 'mi_mi', 'lnc_m', 'm_m','m_mi'])

    parser.add_argument('--memory_capacity', type=int, default=100000)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--target_replace_iter', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--maxiter', type=int, default=5000)
    parser.add_argument('--path', type=str, default='../data/example_network')
    parser.add_argument('--param_path', type=str, default='./save_model/9999_-13.pth')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--step_size', type=int, default=10000)
    parser.add_argument('--learning_rate_gamma', type=float, default=1.0)

    args = parser.parse_args().__dict__
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed = 1000
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    torch.cuda.set_device(0)
    action_choosed=model_sort()