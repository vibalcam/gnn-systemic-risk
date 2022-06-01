import torch
import torch.nn as nn


# class NormModule(nn.Module):
#     def __init__(self, norm_type:str, **kwargs) -> None:
#         """
#         Wrapper normalization module.

#         Input for `BatchNorm` or `GraphNorm` are the node features

#         Input for `EdgeWeightNorm` are the edge weights

#         :param norm_type: `bn` for `BatchNorm`; `gn` for `GraphNorm`; other for `EdgeWeightNorm`; `all` for `GraphNorm` and `EdgeWeightNorm`
#         :param kwargs: other arguments for the corresponding normalization
#         """
#         super().__init__()
#         self.norm_type = norm_type
#         if norm_type=='all':
#             self.edge_type = None
#             self.norm = (GraphNorm('gn', kwargs), dglnn.EdgeWeightNorm('both'))
#         if norm_type in ['bn','gn']:
#             self.edge_type = False
#             self.norm = GraphNorm(norm_type, kwargs)
#         else:
#             self.edge_type = True
#             self.norm = dglnn.EdgeWeightNorm(norm_type, kwargs)

#     def forward(self, graph, node_feats=None, edge_weights=None):
#         # distintion is made so when used it is made explicit what is going through the normalization
#         if self.edge_type is None:
#             return self.norm[0](graph, node_feats), self.norm[1](graph, edge_weights)
#         elif self.edge_type:
#             return self.norm(graph, edge_weights)
#         else:
#             return self.norm(graph, node_feats)


class GraphNorm(nn.Module):
    """
    Code under MIT license from: 
    https://github.com/lsj2408/GraphNorm/blob/master/GraphNorm_ws/gnn_ws/gnn_example/model/Norm/norm.py

    A GraphNorm layer implementation from

    GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training (arXiv:2009.03294v1)
        Tianle Cai, Shengjie Luo, Keyulu Xu, Di He, Tie-yan Liu, Liwei Wang
    """

    def __init__(self, norm_type, hidden_dim=64):
        super(GraphNorm, self).__init__()
        # assert norm_type in ['bn', 'ln', 'gn', None]
        self.norm = None
        if norm_type == 'bn':
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == 'gn':
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, graph, tensor):
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes().cpu().numpy()
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias
