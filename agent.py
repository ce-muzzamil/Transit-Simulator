import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from ray.rllib.utils.annotations import override


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


class Actor(nn.Module):
    def __init__(self, feature_dim: int, num_actions: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_actions = num_actions

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, self.num_actions),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class Critic(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


# 1. Define your Feature Extractor
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, gnn_hidden_dim, embed_size):
        super().__init__()
        self.fc = nn.Linear(input_dim, embed_size)
        
    def forward(self, obs):
        return torch.relu(self.fc(obs))

# 2. Define GNNPolicy without space arguments in __init__
class GNNPolicy(TorchRLModule):
    def __init__(self, config):
        super().__init__(config)
        self.config = config  # Stores all config including spaces
        
    def setup(self):
        model_config = self.config.get("model_config", {})
        input_dim = self.config.observation_space.shape[0]
        
        self.feature_extractor = FeatureExtractor(
            input_dim=input_dim,
            gnn_hidden_dim=model_config.get("gnn_hidden_dim", 128),
            embed_size=model_config.get("embed_size", 256)
        )
        self.actor = nn.Linear(model_config["embed_size"], self.config.action_space.n)
        self.critic = nn.Linear(model_config["embed_size"], 1)

    def forward_train(self, batch):
        features = self.feature_extractor(batch["obs"])
        return {
            "logits": self.actor(features),
            "vf_preds": self.critic(features).squeeze(-1)
        }

    def forward_inference(self, batch):
        return self.forward_train(batch)

class SharedGNNMultiAgentModule(MultiRLModule):
    def setup(self):
        module_specs = self.config.get("modules", {})
        
        if not module_specs:
            raise ValueError("No module specs found in config!")
        
        first_module_id = next(iter(module_specs.keys()))
        shared_spec = module_specs[first_module_id]
        shared_module = shared_spec.build()
        
        self._rl_modules = {
            module_id: shared_module
            for module_id in module_specs.keys()
        }
        
        self._mapping = {
            module_id: first_module_id
            for module_id in module_specs.keys()
        }

    def get_module(self, module_id):
        return self._rl_modules.get(module_id)

    def add_module(self, module_id, module):
        self._rl_modules[module_id] = module