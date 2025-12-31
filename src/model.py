import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree, to_dense_batch
from torch_geometric.nn import GraphConv
import torch
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool


class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_weight=True, use_init=False):
        super(GraphConvLayer, self).__init__()

        self.use_init = use_init
        self.use_weight = use_weight
        if self.use_init:
            in_channels_ = 2 * in_channels
        else:
            in_channels_ = in_channels
        self.W = nn.Linear(in_channels_, out_channels)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x, edge_index, x0, edge_weight=None):
        N = x.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm_in = (1. / d[col]).sqrt()
        d_norm_out = (1. / d[row]).sqrt()

        if edge_weight is None:
            edge_weight = torch.ones_like(row).float()

        value = edge_weight * d_norm_in * d_norm_out
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        x = matmul(adj, x)  # [N, D]

        if self.use_init:
            x = torch.cat([x, x0], 1)
            x = self.W(x)
        elif self.use_weight:
            x = self.W(x)

        return x


class GraphConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.5, use_bn=True, use_residual=True,
                 use_weight=True, use_init=False, use_act=True):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers):
            self.convs.append(
                GraphConvLayer(hidden_channels, hidden_channels, use_weight, use_init))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        layer_ = []

        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, layer_[0], edge_weight=edge_weight)
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + layer_[-1]
        return x


# GC-GNN
class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=7, dropout=0.5):
        super(GNN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_channels, hidden_channels, num_layers=1))
        for _ in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_channels, hidden_channels, num_layers=1))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ========= 自定义稳定 Transformer Layer =========
class StableTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

        self.pos_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))

    # self.pos_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        # Attention + 残差 + LN
        key_padding_mask = ~mask if mask is not None else None

        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        x = torch.clamp(x, -1e4, 1e4)  # 防止爆炸

        # FFN + 残差 + LN
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        x = torch.clamp(x, -1e4, 1e4)

        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

        return x


# ========= GRL融合模块 =========
class GRLFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, gnn_feat, trans_feat):
        combined = torch.cat([gnn_feat, trans_feat], dim=-1)
        gate = self.mlp(combined)
        fused = gnn_feat + gate * trans_feat  # 残差融合
        return fused


# ========= GFormer with GRL =========
class GFormer(nn.Module):
    def __init__(self, node_features_dim, num_classes, hidden_channels,
                 trans_num_layers=2, trans_num_heads=4, trans_dropout=0.1,
                 gnn_num_layers=7, gnn_dropout=0.5):
        super().__init__()

        self.graph_conv = GNN(node_features_dim, hidden_channels,
                              num_layers=gnn_num_layers, dropout=gnn_dropout)

        self.input_proj = nn.Sequential(
            nn.Linear(node_features_dim, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels)
        )

        self.trans_layers = nn.Sequential(*[
            StableTransformerLayer(hidden_channels, trans_num_heads, trans_dropout)
            for _ in range(trans_num_layers)
        ])

        self.grl_fusion = GRLFusion(hidden_channels)

        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels * 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels * 2, num_classes)
        )

    def forward(self, x, edge_index, batch, edge_weight=None, edge_type=None, pos=None, node_type=None):
        # 输入安全处理
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

        # GNN 路径
        gnn_features = self.graph_conv(x, edge_index, edge_weight)

        # Transformer 路径
        x_dense, mask = to_dense_batch(x, batch)
        trans_input = self.input_proj(x_dense)

        trans_output = trans_input
        for layer in self.trans_layers:
            trans_output = layer(trans_output, mask=mask)

        trans_output = trans_output[mask]

        fused_node_features = self.grl_fusion(gnn_features, trans_output)

        graph_level_features = torch.cat([
            global_add_pool(fused_node_features, batch),
            global_max_pool(fused_node_features, batch),
            global_mean_pool(fused_node_features, batch)
        ], dim=-1)

        out = self.output_mlp(graph_level_features)

        return out
