import math

import mindspore as ms
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import initializer, Normal, Zero


def init_params(module, n_layers):
    if isinstance(module, nn.Dense):
        module.weight = Parameter(initializer(
            Normal(sigma=0.02 / math.sqrt(n_layers), mean=0.0),
            shape=module.weight.shape,
            dtype=ms.float32
        ))
        if module.bias is not None:
            module.bias = Parameter(initializer(
                Zero(), shape=module.bias.shape, dtype=ms.float32
            ))
    if isinstance(module, nn.Embedding):
        module.embedding_table = Parameter(initializer(
            Normal(sigma=0.02, mean=0.0),
            shape=module.embedding_table.shape,
            dtype=ms.float32
        ))


class GraphNodeFeature(nn.Cell):
    """
        Compute node features for each node in the graph.
    """

    def __init__(self, num_heads, num_atoms, num_in_degree, num_out_degree, hidden_dim, n_layers):
        super(GraphNodeFeature, self).__init__()
        self.num_heads = num_heads
        self.num_atoms = num_atoms

        # 1 for graph token
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(
            num_out_degree, hidden_dim, padding_idx=0
        )

        self.graph_token = nn.Embedding(1, hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def construct(self, x, in_degree, out_degree):
        # x, in_degree, out_degree = (
        #     batched_data["x"].astype(ms.int64),
        #     batched_data["in_degree"].astype(ms.int64),
        #     batched_data["out_degree"].astype(ms.int64),
        # )
        print(x)
        x = x.astype(ms.int64)
        in_degree = in_degree.astype(ms.int64)
        out_degree = out_degree.astype(ms.int64)

        n_graph, n_node = x.shape[:2]

        node_feature = self.atom_encoder(x).sum(axis=-2)

        node_feature = (
                node_feature
                + self.in_degree_encoder(in_degree)
                + self.out_degree_encoder(out_degree)
        )

        graph_token_feature = self.graph_token.embedding_table.unsqueeze(0).tile((n_graph, 1, 1))

        graph_node_feature = ops.cat([graph_token_feature, node_feature], axis=1)

        return graph_node_feature


class GraphAttnBias(nn.Cell):
    """
    Compute attention bias for each head.
    """

    def __init__(
            self,
            num_heads,
            num_atoms,
            num_edges,
            num_spatial,
            num_edge_dis,
            hidden_dim,
            edge_type,
            multi_hop_max_dist,
            n_layers,
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist

        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        self.edge_type = edge_type
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(num_edge_dis * num_heads * num_heads, 1)

        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def construct(self, batched_data):
        attn_bias, spatial_pos, x = (
            batched_data["attn_bias"],
            batched_data["spatial_pos"].astype(ms.int64),
            batched_data["x"],
        )
        # in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
        edge_input, attn_edge_type = (
            batched_data["edge_input"],
            batched_data["attn_edge_type"].astype(ms.int64),
        )

        n_graph, n_node = x.shape[:2]
        graph_attn_bias = attn_bias.copy()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).tile((1, self.num_heads, 1, 1))
        # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # reset spatial pos here
        t = self.graph_token_virtual_distance.embedding_table.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        if self.edge_type == "multi_hop":
            spatial_pos_ = spatial_pos.copy()
            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, x > 1 to x - 1
            spatial_pos_ = ops.where(
                condition=spatial_pos_ > 1,
                x=spatial_pos_ - 1,
                y=spatial_pos_
            )
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]
            # [n_graph, n_node, n_node, max_dist, n_head]
            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.shape[-2]
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape((max_dist, -1, self.num_heads))
            edge_input_flat = ops.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads)[:max_dist, :, :],
            )
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads
            ).permute(1, 2, 3, 0, 4)
            edge_input = (
                edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))
            ).permute(0, 3, 1, 2)
        else:
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            edge_input = self.edge_encoder(attn_edge_type).mean(-2, keep_dims=True).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        return graph_attn_bias















