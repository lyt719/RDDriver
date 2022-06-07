import numpy as np
import networkx as nx
import dgl
from load import *
from sklearn import preprocessing
import sympy as sp
import torch
import os

class MVC_env:
    def __init__(self,  replay_penalty=0):
        self.replay_penalty = replay_penalty

    def reset(self,g):
        self.graph= g
        self.number_nodes=self.graph.number_of_nodes()
        self.rna_num = {}
        self.num_rna={}
        curnode=0
        for name in self.graph.ntypes:
            tmp={}
            for i in range(self.graph.number_of_nodes(name)):
                tmp[i]=curnode
                self.num_rna[curnode]=(name,i)
                curnode+=1
            self.rna_num[name]=tmp
        self.state= torch.zeros(self.number_nodes)
        self.adjacency_matrix=torch.from_numpy(np.array(nx.adjacency_matrix(dgl.to_homogeneous(self.graph).to_networkx().to_undirected()).todense(),dtype=float)).to('cuda')
        return self.graph,self.state.clone(),self.num_rna


    def PBH(self):
        action_index = [i for i, x in enumerate(self.state) if x == 1]
        B = []
        for i in action_index:
            B_part = [0] * self.number_nodes
            B_part[i] = 1
            B.append(B_part) # B
        B=torch.from_numpy(np.array(B,dtype='float')).to('cuda')
        IA = - self.adjacency_matrix
        rank = torch.matrix_rank(torch.cat((IA, B.t()), dim=1))
        return rank



    def is_done(self):
        if self.PBH()==self.number_nodes:
            return True
        else:
            return False


    def step(self, action,test=False):
        if self.state[action] != 1:
            self.state[action] = 1
            rew=-1
        else:
            rew = -self.replay_penalty
        if test:
            return self.state.clone(),rew,False
        else:
            return self.state.clone(), rew, self.is_done()