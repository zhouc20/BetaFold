import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.register import register_head
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, global_max_pool

from layer.GPS_layer import GPSLayer
from layer.GPS_equiformer_layer import GPS_equiformer_Layer
from encoder.Encoder import Amino_Acid, Atom_encoder, Bond_encoder


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


# @register_network('GPSModel')
# class GPSModel(torch.nn.Module):
#     """Multi-scale graph x-former.
#     """
#
#     def __init__(self, dim_in, dim_out):
#         super().__init__()
#         self.encoder = FeatureEncoder(dim_in)
#         dim_in = self.encoder.dim_in
#
#         if cfg.gnn.layers_pre_mp > 0:
#             self.pre_mp = GNNPreMP(
#                 dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
#             dim_in = cfg.gnn.dim_inner
#
#         assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
#             "The inner and hidden dims must match."
#
#         try:
#             local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
#         except:
#             raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
#         layers = []
#         for _ in range(cfg.gt.layers):
#             layers.append(GPSLayer(
#                 dim_h=cfg.gt.dim_hidden,
#                 local_gnn_type=local_gnn_type,
#                 global_model_type=global_model_type,
#                 num_heads=cfg.gt.n_heads,
#                 act=cfg.gnn.act,
#                 pna_degrees=cfg.gt.pna_degrees,
#                 equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
#                 dropout=cfg.gt.dropout,
#                 attn_dropout=cfg.gt.attn_dropout,
#                 layer_norm=cfg.gt.layer_norm,
#                 batch_norm=cfg.gt.batch_norm,
#                 bigbird_cfg=cfg.gt.bigbird,
#                 log_attn_weights=cfg.train.mode == 'log-attn-weights'
#             ))
#         self.layers = torch.nn.Sequential(*layers)
#
#         GNNHead = register.head_dict[cfg.gnn.head]
#         self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)
#
#     def forward(self, batch):
#         for module in self.children():
#             batch = module(batch)
#         return batch


@register_network('GPSModel')
class GPSModel(torch.nn.Module):
    """Multi-scale graph x-former.
    """

    def __init__(self, hid_dim, out_dim, num_layers, num_layers_regression, num_layers_cls, global_pool,
                 local_gnn_type, global_model_type, num_heads, act='relu',
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, log_attn_weights=False, max_length=512,
                 equiformer=False, radius=5.0, max_num_neighbors=256):
        super().__init__()
        self.acid_encoder = Amino_Acid(hid_dim)
        self.bond_encoder = Bond_encoder(hid_dim)

        layers = []
        for _ in range(num_layers):
            layers.append(GPS_equiformer_Layer(
                dim_h=hid_dim,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=num_heads,
                act=act,
                pna_degrees=pna_degrees,
                equivstable_pe=equivstable_pe,
                dropout=dropout,
                attn_dropout=attn_dropout,
                layer_norm=layer_norm,
                batch_norm=batch_norm,
                bigbird_cfg=bigbird_cfg,
                log_attn_weights=log_attn_weights,
                max_length=max_length if _ == 0 else None,
                equiformer=equiformer,
                max_num_neighbors=max_num_neighbors,
                radius=radius
            ))
        self.layers = torch.nn.Sequential(*layers)

        self.cls = ClassifierHead(dim_in=hid_dim, dim_out=26, L=num_layers_cls)
        self.post_mp = SANGraphHead(dim_in=hid_dim, dim_out=out_dim, L=num_layers_regression, global_pool=global_pool)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class ClassifierHead(nn.Module):
    def __init__(self, dim_in, dim_out=26, L=0):
        super(ClassifierHead, self).__init__()
        list_FC_layers = [
            nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True)
            for l in range(L)]
        list_FC_layers.append(
            nn.Linear(dim_in // 2 ** L, dim_out, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = nn.ReLU(inplace=True)

    def forward(self, batch):
        graph_emb = batch.x
        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation(graph_emb)
        logits = self.FC_layers[self.L](graph_emb)
        batch.logits = logits
        return batch


@register_head('san_graph')
class SANGraphHead(nn.Module):
    """
    SAN prediction head for graph prediction tasks.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        L (int): Number of hidden layers.
    """

    def __init__(self, dim_in, dim_out, L=2, global_pool='add'):
        super().__init__()
        # self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]
        assert global_pool in ['add', 'mean', 'max']
        self.pooling_fun = eval("global_" + global_pool + "_pool")
        list_FC_layers = [
            nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True)
            for l in range(L)]
        list_FC_layers.append(
            nn.Linear(dim_in // 2 ** L, dim_out, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = nn.ReLU(inplace=True)

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y, batch.logits

    def forward(self, batch):
        graph_emb = self.pooling_fun(batch.x, batch.batch)
        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation(graph_emb)
        graph_emb = self.FC_layers[self.L](graph_emb)
        batch.graph_feature = graph_emb
        pred, label, logits = self._apply_index(batch)
        return pred, label, logits



