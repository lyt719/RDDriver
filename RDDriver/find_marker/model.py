import torch.nn.functional as F
from dgl.nn.pytorch import HeteroGraphConv, GraphConv, WeightBasis
from layers import *
import torch
import torch.nn as nn

class RGCNDQNModel(nn.Module):
    def __init__(self, feat_dim, in_size, out_size,rel_names, num_bases, num_hidden_layers):
        super(RGCNDQNModel, self).__init__()
        self.feat_dim = feat_dim
        self.in_size = in_size
        self.out_size = out_size
        self.rel_names = rel_names
        self.num_relations = int(len(rel_names) / 2 - 1)
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers

        self.encoder_layers = nn.ModuleList()

        self.encoder_layers.append(RelGraphConvLayer(
            self.in_size, self.feat_dim, self.rel_names, self.num_bases, activation=F.relu))
        for i in range(self.num_hidden_layers):
            self.encoder_layers.append(RelGraphConvLayer(
                self.feat_dim, self.feat_dim, self.rel_names, self.num_bases, activation=F.relu))

        self.decoder_layers = Net(self.feat_dim,self.out_size )

    def forward(self, args, g, h=None, blocks=None):
        if h is None:
            f_mirna = torch.zeros(g.number_of_nodes('mirna'), 1).to(args['device'])
            f_mrna = torch.zeros(g.number_of_nodes('mrna'), 1).to(args['device'])
            f_lncrna = torch.zeros(g.number_of_nodes('lncrna'), 1).to(args['device'])
            h = {'lncrna': f_lncrna,'mirna': f_mirna, 'mrna': f_mrna}

        if blocks is None:
            for layer in self.encoder_layers:
                h = layer(g.to(args['device']), h)
        else:
            for layer, block in zip(self.encoder_layers, blocks):
                h = layer(block, h)


        return torch.cat((self.decoder_layers(h['lncrna']),self.decoder_layers(h['mirna']),self.decoder_layers(h['mrna'])),dim=0)


