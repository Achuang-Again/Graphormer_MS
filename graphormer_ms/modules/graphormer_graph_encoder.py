from typing import Optional, Tuple

from mindspore import ops, nn, Tensor, Parameter
import mindspore as ms

from fairseq.modules import FairseqDropout, LayerNorm, LayerDropModuleList
from fairseq.modules import quant_noise as apply_quant_noise_

from .multihead_attention import MultiheadAttention
from .graphormer_layers import GraphNodeFeature, GraphAttnBias
from .graphormer_graph_encoder_layer import GraphormerGraphEncoderLayer


def init_graphormer_params(module):
    """
        Initialize the weights specific to the Graphormer Model.
    """
    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data = ops.normal(shape=data.shape, mean=0.0, stddev=0.02)
        return data

    if isinstance(module, nn.Dense):
        module.weight = Parameter(normal_(module.weight))
        if module.bias is not None:
            module.bias = Parameter(
                ops.zeros(size=module.bias.shape, dtype=ms.float32))
    if isinstance(module, nn.Embedding):
        module.weight = Parameter(normal_(module.weight))
        if module.padding_idx is not None:
            module.weight[module.padding_idx] = \
                ops.zeros(size=module.weight[module.padding_idx].shape,
                          dtype=ms.float32)
    if isinstance(module, MultiheadAttention):
        module.q_proj.module.weight = Parameter(normal_(module.q_proj.module.weight))
        module.k_proj.module.weight = Parameter(normal_(module.k_proj.module.weight))
        module.v_proj.module.weight = Parameter(normal_(module.v_proj.module.weight))
        

class GraphormerGraphEncoder(nn.Cell):
    def __init__(
            self,
            num_atoms: int,
            num_in_degree: int,
            num_out_degree: int,
            num_edges: int,
            num_spatial: int,
            num_edge_dis: int,
            edge_type: str,
            multi_hop_max_dist: int,
            num_encoder_layers: int = 12,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 768,
            num_attention_heads: int = 32,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            layerdrop: float = 0.0,
            encoder_normalize_before: bool = False,
            pre_layernorm: bool = False,
            apply_graphormer_init: bool = False,
            activation_fn: str = "gelu",
            embed_scale: float = None,
            freeze_embeddings: bool = False,
            n_trans_layers_to_freeze: int = 0,
            export: bool = False,
            traceable: bool = False,
            q_noise: float = 0.0,
            qn_block_size: int = 8,
    ) -> None:
        super().__init__()
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = layerdrop
        self.embedding_dim = embedding_dim
        self.apply_graphormer_init = apply_graphormer_init
        self.traceable = traceable

        self.graph_node_feature = GraphNodeFeature(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
        )

        self.graph_attn_bias = GraphAttnBias(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
        )

        self.embed_scale = embed_scale

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Dense(self.embedding_dim, self.embedding_dim, has_bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        if pre_layernorm:
            self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.CellList([])
        self.layers.extend(
            [
                self.build_graphormer_graph_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    export=export,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                    pre_layernorm=pre_layernorm,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # Apply initialization of model params after building the model
        if self.apply_graphormer_init:
            self.apply(init_graphormer_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.get_parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            raise NotImplementedError("Freezing embeddings is not implemented yet.")

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])


    def build_graphormer_graph_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        export,
        q_noise,
        qn_block_size,
        pre_layernorm,
    ):
        return GraphormerGraphEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            pre_layernorm=pre_layernorm,
        )


    def construct(
        self,
        batched_data,
        perturb=None,
        last_state_only: bool = False,
        token_embeddings: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:

        is_tpu = False
        # compute padding mask. This is needed for multi-head attention
        data_x = batched_data["x"]
        n_graph, n_node = data_x.shape[:2]
        padding_mask = (data_x[:, :, 0]).equal(0)
        padding_mask_cls = ops.zeros((n_graph, 1), dtype=padding_mask.dtype)
        padding_mask = ops.cat((padding_mask_cls, padding_mask), axis=1)
        # B x (T+1) x 1

        if token_embeddings is not None:
            x = token_embeddings
        else:
            x = self.graph_node_feature(batched_data["x"], batched_data["in_degree"], batched_data["out_degree"])

        if perturb is not None:
            x[:, 1:, :] += perturb

        # x: B x T x C

        attn_bias = self.graph_attn_bias(batched_data)

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation

        # B x T x C -> T x B x C
        input_perm = list(range(x.ndim))
        input_perm[0], input_perm[1] = input_perm[1], input_perm[0]
        x = x.transpose(input_perm)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, _ = layer(
                x,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
            )
            if not last_state_only:
                inner_states.append(x)

            graph_rep = x[0, :, :]

            if last_state_only:
                inner_states = [x]

            if self.traceable:
                return ops.stack(inner_states), graph_rep
            else:
                return inner_states, graph_rep













