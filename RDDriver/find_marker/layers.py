import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import HeteroGraphConv, GraphConv, WeightBasis
import numpy as np

class RelGraphConvLayer(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=True,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = HeteroGraphConv({
            rel: GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
            for rel in rel_names}, aggregate='sum')

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))


        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)


        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        #local_var：The returned graph object shares the feature data and graph structure of this graph. However, any out-place mutation to the feature data will not reflect to this graph, thus making it easier to use in a function scope
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                    for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}
        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs
        hs = self.conv(g, inputs, mod_kwargs=wdict)    # dict

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class Net(nn.Module):
    def __init__(self,feat_dim,o_dim=1 ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(feat_dim, feat_dim//2)
        nn.init.kaiming_normal_(self.fc1.weight.data)
        self.out = nn.Linear(feat_dim//2, o_dim)
        nn.init.kaiming_normal_(self.out.weight.data)
        self.ReLU = nn.ReLU()
        self.LeakyReLU=nn.LeakyReLU()

    def forward(self, x):
        return self.LeakyReLU(self.out(self.ReLU(self.fc1(x))))
