import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core import Columns
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule


class GATv2FeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels,
        edge_dim,
        hidden_dim=128,
        num_heads=4,
        out_dim=256,
        dropout_rate=0.0,
    ):
        super().__init__()

        self.gat1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout_rate,
        )

        self.gat2 = GATv2Conv(
            in_channels=hidden_dim * num_heads,
            out_channels=out_dim,
            heads=1,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout_rate,
        )

        self.dropout = nn.Dropout(0.1)

    def process_for_gat(self, gat, x, edge_index, edge_attr):
        if x.ndim == 2:
            return gat(x, edge_index, edge_attr)
        else:
            N = x.shape[0]
            outs = []
            for i in range(N):
                outs.append(gat(x[i], edge_index[i], edge_attr[i]))
            return torch.stack(outs, dim=0)

    def forward(self, data):
        if "batch" not in data.keys():
            batch = None
            x, edge_index, edge_attr = (
                data["x"],
                data["edge_index"].long(),
                data["edge_attr"],
            )
        else:
            x, edge_index, edge_attr, batch = (
                data["x"],
                data["edge_index"].long(),
                data["edge_attr"],
                data["batch"],
            )

        if torch.isnan(x).any():
            print("Found NaNs in obs.x")
        if torch.isnan(edge_attr).any():
            print("Found NaNs in obs.edge_attr")
        if torch.isnan(edge_index).any():
            print("Found NaNs in obs.edge_index")

        x = self.process_for_gat(self.gat1, x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.process_for_gat(self.gat2, x, edge_index, edge_attr)
        x = torch.relu(x)  # N,L,E

        return x


class EncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_rate=0.0):
        super().__init__()

        self.mha = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.norm_1 = nn.LayerNorm(embed_size)
        self.dropout_1 = nn.Dropout(dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )

        self.norm_2 = nn.LayerNorm(embed_size)
        self.dropout_2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.mha(x, x, x)
        x = self.norm_1(x + self.dropout_1(attn_output))

        # Feed Forward Network
        ffn_output = self.ffn(x)
        x = self.norm_2(x + self.dropout_2(ffn_output))

        return x


class DecoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_rate=0.0):
        super().__init__()

        # Masked Self-attention
        self.self_mha = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.norm_1 = nn.LayerNorm(embed_size)
        self.dropout_1 = nn.Dropout(dropout_rate)

        # Cross-attention with encoder output
        self.cross_mha = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.norm_2 = nn.LayerNorm(embed_size)
        self.dropout_2 = nn.Dropout(dropout_rate)

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )

        self.norm_3 = nn.LayerNorm(embed_size)
        self.dropout_3 = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output):
        # Masked Self-Attention (future masking)
        self_attn_output, _ = self.self_mha(x, x, x)
        x = self.norm_1(x + self.dropout_1(self_attn_output))

        # Cross-Attention (attend to encoder output)
        cross_attn_output, _ = self.cross_mha(x, enc_output, enc_output)
        x = self.norm_2(x + self.dropout_2(cross_attn_output))

        # Feed Forward Network
        ffn_output = self.ffn(x)
        x = self.norm_3(x + self.dropout_3(ffn_output))

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        embed_size,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_rate=0.0,
    ):

        super().__init__()

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(embed_size, num_heads, dropout_rate)
                for _ in range(num_encoder_layers)
            ]
        )

        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(embed_size, num_heads, dropout_rate)
                for _ in range(num_decoder_layers)
            ]
        )

    def forward(self, src, tgt):
        # src: (N, L_src, E)
        # tgt: (N, L_tgt, E)

        enc_output = src
        for layer in self.encoder_layers:
            enc_output = layer(enc_output)

        dec_output = tgt
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output)

        return dec_output


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        observation_space,
        gnn_hidden_dim=128,
        gnn_num_heads=4,
        embed_size=256,
        transformer_num_heads=4,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout_rate=0.0,
    ):
        super().__init__()

        self.feature_dim = embed_size
        self.topology = GATv2FeatureExtractor(
            observation_space["x"].shape[-1],
            observation_space["edge_attr"].shape[-1],
            gnn_hidden_dim,
            gnn_num_heads,
            embed_size,
            dropout_rate=dropout_rate,
        )

        self.route = GATv2FeatureExtractor(
            observation_space["x_route"].shape[-1],
            observation_space["edge_attr_route"].shape[-1],
            gnn_hidden_dim,
            gnn_num_heads,
            embed_size,
            dropout_rate=dropout_rate,
        )

        self.transformer = Transformer(
            embed_size=embed_size,
            num_heads=transformer_num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout_rate=dropout_rate,
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )

    def forward(self, observations):
        route = {
            f"x": observations[f"x_route"],
            f"edge_index": observations[f"edge_index_route"],
            f"edge_attr": observations[f"edge_attr_route"],
        }

        observations = {
            "x": observations["x"],
            "edge_index": observations["edge_index"],
            "edge_attr": observations["edge_attr"],
        }

        topology_vector = self.topology(observations)  # N,L,E
        routes_vector = self.route(route)  # (N, L, E)

        if routes_vector.ndim == 2:
            routes_vector = routes_vector.unsqueeze(0)
        if topology_vector.ndim == 2:
            topology_vector = topology_vector.unsqueeze(0)
        out = self.transformer(topology_vector, routes_vector)  # N,L,E
        out = torch.mean(out, dim=1)  # N,E
        out = self.ffn(out)  # N,E
        return out


class GNNPolicy(TorchRLModule):
    def setup(self):
        super().setup()

        self.feature_extractor = FeatureExtractor(
            input_dim=self.config.observation_space,
            gnn_hidden_dim=self.model_config.get("gnn_hidden_dim", 128),
            gnn_num_heads=self.model_config.get("gnn_num_heads", 4),
            embed_size=self.model_config.get("embed_size", 256),
            transformer_num_heads=self.model_config.get("transformer_num_heads", 4),
            num_encoder_layers=self.model_config.get("num_encoder_layers", 6),
            num_decoder_layers=self.model_config.get("num_decoder_layers", 6),
            dropout_rate=self.model_config.get("dropout_rate", 0.0),
        )

        self._net = (
            nn.Linear(
                self.model_config.get("embed_size", 256),
                self.model_config.get("embed_size", 256),
            ),
        )

    def _forward(self, batch, **kwargs):
        embeddings = self._net(self.feature_extractor(batch[Columns.OBS]))
        return {"encoder_embeddings": embeddings}


class Policy(TorchRLModule):
    def setup(self):
        super().setup()

        embedding_dim = self.model_config["embedding_dim"]
        hidden_dim = self.model_config["hidden_dim"]

        self._pi_head = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, self.action_space.n),
        )

    def _forward(self, batch, **kwargs):
        embeddings = batch["encoder_embeddings"]
        logits = self._pi_head(embeddings)
        return {Columns.ACTION_DIST_INPUTS: logits}


class SharedGNNMultiAgentModule(MultiRLModule):
    def setup(self):
        super().setup()
        assert (
            "gnn-policy" in self._rl_modules
            and isinstance(self._rl_modules["gnn-policy"], GNNPolicy)
            and len(self._rl_modules) > 1
        )
        self.encoder = self._rl_modules["gnn-policy"]

    def _forward(self, batch, **kwargs):
        outputs = {}

        # Loop through the policy nets (through the given batch's keys).
        for policy_id, policy_batch in batch.items():
            rl_module = self._rl_modules[policy_id]

            # Pass policy's observations through shared encoder to get the features for
            # this policy.
            policy_batch["encoder_embeddings"] = self.encoder._forward(batch[policy_id])

            # Pass the policy's embeddings through the policy net.
            outputs[policy_id] = rl_module._forward(batch[policy_id], **kwargs)

        return outputs
