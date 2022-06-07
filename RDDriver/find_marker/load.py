import random
import dgl
import torch
import numpy as np
import argparse
import pandas as pd


def load_random_hetero_graph(path='../data/random_network',Specified_quantity={} ):
    graph_dict={}
    a=pd.read_csv(path+'/mi_lnc.csv')
    graph_dict[('mirna', 'mi_lnc', 'lncrna')]=(torch.tensor(a['mirna']), torch.tensor(a['lncrna']))
    graph_dict[('lncrna', 'lnc_mi', 'mirna')]=(torch.tensor(a['lncrna']),torch.tensor(a['mirna']))
    a = pd.read_csv(path+'/mi_m.csv')
    graph_dict[('mirna', 'mi_m', 'mrna')] = (torch.tensor(a['mirna']), torch.tensor(a['mrna']))
    graph_dict[('mrna', 'm_mi', 'mirna')] = ( torch.tensor(a['mrna']),torch.tensor(a['mirna']))
    a = pd.read_csv(path+'/m_lnc.csv')
    graph_dict[('lncrna', 'm_lnc', 'mrna')] = (torch.tensor(a['lncrna']), torch.tensor(a['mrna']))
    graph_dict[('mrna', 'lnc_m', 'lncrna')] = (torch.tensor(a['mrna']), torch.tensor(a['lncrna']))

    a = pd.read_csv(path+'/mi_mi.csv')
    graph_dict[('mirna', 'mi_mi', 'mirna')] = (torch.tensor(a['mirna1'].tolist()+a['mirna2'].tolist()), torch.tensor(a['mirna2'].tolist()+a['mirna1'].tolist()))
    a = pd.read_csv(path+'/m_m.csv')
    graph_dict[('mrna', 'm_m', 'mrna')] = (torch.tensor(a['mrna1'].tolist()+a['mrna2'].tolist()), torch.tensor(a['mrna2'].tolist()+a['mrna1'].tolist()))
    a = pd.read_csv(path+'/lnc_lnc.csv')
    graph_dict[('lncrna', 'lnc_lnc', 'lncrna')] = (torch.tensor(a['lncrna1'].tolist()+a['lncrna2'].tolist()), torch.tensor(a['lncrna2'].tolist()+a['lncrna1'].tolist()))

    if len(Specified_quantity)==0:
        g = dgl.heterograph(graph_dict)
    else:
        g = dgl.heterograph(graph_dict,num_nodes_dict=Specified_quantity)
    return g

