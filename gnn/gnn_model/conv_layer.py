import sys
sys.path.append('../../')

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import degree, softmax
from torch_geometric.nn import MessagePassing

from modules.encoder import BondEncoder


class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), 
                                       torch.nn.BatchNorm1d(2*emb_dim), 
                                       torch.nn.ReLU(), 
                                       torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr, edge_atten = None):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding, edge_atten=edge_atten))

        return out

    def message(self, x_j, edge_attr, edge_atten = None):
        m = F.relu(x_j + edge_attr)
        
        if edge_atten is not None:
            m = m * edge_atten
        
        return m

    def update(self, aggr_out):
        return aggr_out


class GATConv(MessagePassing):
    def __init__(
        self,
        emb_dim,
        heads=1,
        dropout=0.0,
        negative_slope=0.2,
        residual=False,
    ):
        """
        GAT convolution with bond / edge features.

        emb_dim: node embedding dimensionality
        heads: number of attention heads
        dropout: attention dropout
        negative_slope: LeakyReLU slope for attention logits
        residual: whether to add root/residual node representation
        """

        super(GATConv, self).__init__(aggr="add", node_dim = 0)

        assert emb_dim % heads == 0, "emb_dim must be divisible by heads"

        self.emb_dim = emb_dim
        self.heads = heads
        self.out_dim = emb_dim // heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.residual = residual

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_linear = torch.nn.Linear(emb_dim, emb_dim)

        self.att_src = torch.nn.Parameter(torch.Tensor(1, heads, self.out_dim))
        self.att_dst = torch.nn.Parameter(torch.Tensor(1, heads, self.out_dim))

        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

        if residual:
            self.root_linear = torch.nn.Linear(emb_dim, emb_dim)
            self.root_emb = torch.nn.Embedding(1, emb_dim)

    def forward(self, x, edge_index, edge_attr, edge_atten=None):
        x_input = x

        x = self.linear(x)
        x = x.view(-1, self.heads, self.out_dim)

        edge_embedding = self.bond_encoder(edge_attr)
        edge_embedding = self.edge_linear(edge_embedding)
        edge_embedding = edge_embedding.view(-1, self.heads, self.out_dim)

        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_embedding,
            edge_atten=edge_atten,
        )

        out = out.view(-1, self.emb_dim)

        if self.residual:
            out = out + F.relu(
                self.root_linear(x_input) + self.root_emb.weight
            )

        return out

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i, edge_atten=None):
        # edge-aware message
        msg = F.relu(x_j + edge_attr)

        # attention score: target node x_i attends to source message msg
        alpha = (x_i * self.att_dst).sum(dim=-1) + (msg * self.att_src).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=self.negative_slope)

        # normalize attention weights over incoming edges
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = msg * alpha.unsqueeze(-1)

        if edge_atten is not None:
            if edge_atten.dim() == 1:
                edge_atten = edge_atten.view(-1, 1, 1)
            elif edge_atten.dim() == 2:
                edge_atten = edge_atten.unsqueeze(-1)

            msg = msg * edge_atten

        return msg

    def update(self, aggr_out):
        return aggr_out