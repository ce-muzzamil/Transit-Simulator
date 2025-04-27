import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class GATv2FeatureExtractor(nn.Module):
    def __init__(self, observation_space, 
                 hidden_dim=128, 
                 num_heads=4, 
                 out_dim=256,
                 dropout_rate=0.0):
        super().__init__()

        in_channels = observation_space["x"].shape[1]
        edge_dim = observation_space["edge_attr"].shape[-1]

        self.gat1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout_rate
        )

        self.gat2 = GATv2Conv(
            in_channels=hidden_dim * num_heads,
            out_channels=out_dim,
            heads=1,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout_rate
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
            x, edge_index, edge_attr = data["x"], data["edge_index"].long(), data["edge_attr"]
        else:
            x, edge_index, edge_attr, batch = data["x"], data["edge_index"].long(), data["edge_attr"], data["batch"]

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
        x = torch.relu(x) # N,L,E
        
        return x
    

class EncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_rate=0.0):
        super().__init__()
        
        self.mha = nn.MultiheadAttention(embed_dim=embed_size, 
                                         num_heads=num_heads,
                                         dropout=dropout_rate,
                                         batch_first=True)
        
        self.norm_1 = nn.LayerNorm(embed_size)
        self.dropout_1 = nn.Dropout(dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size)
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
        self.self_mha = nn.MultiheadAttention(embed_dim=embed_size,
                                               num_heads=num_heads,
                                               dropout=dropout_rate,
                                               batch_first=True)

        self.norm_1 = nn.LayerNorm(embed_size)
        self.dropout_1 = nn.Dropout(dropout_rate)

        # Cross-attention with encoder output
        self.cross_mha = nn.MultiheadAttention(embed_dim=embed_size,
                                                num_heads=num_heads,
                                                dropout=dropout_rate,
                                                batch_first=True)
        
        self.norm_2 = nn.LayerNorm(embed_size)
        self.dropout_2 = nn.Dropout(dropout_rate)

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size)
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
    def __init__(self, 
                 embed_size, 
                 num_heads, 
                 num_encoder_layers, 
                 num_decoder_layers, 
                 dropout_rate=0.0):
        
        super().__init__()

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embed_size, num_heads, dropout_rate)
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embed_size, num_heads, dropout_rate)
            for _ in range(num_decoder_layers)
        ])

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


class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space, 
                 gnn_hidden_dim=128, 
                 gnn_num_heads=4, 
                 embed_size=256, 
                 transformer_num_heads=4, 
                 num_encoder_layers=6, 
                 num_decoder_layers=6, 
                 dropout_rate=0.0
                 ):
        super().__init__(observation_space, embed_size)

        self.feature_dim = embed_size
        self.topology = GATv2FeatureExtractor(observation_space, 
                                              gnn_hidden_dim, 
                                              gnn_num_heads, 
                                              embed_size,
                                              dropout_rate=dropout_rate)
        
        self.route = GATv2FeatureExtractor(observation_space, 
                                           gnn_hidden_dim, 
                                           gnn_num_heads, 
                                           embed_size,
                                           dropout_rate=dropout_rate)
        
        self.transformer = Transformer(embed_size=embed_size, 
                                       num_heads=transformer_num_heads, 
                                       num_encoder_layers=num_encoder_layers, 
                                       num_decoder_layers=num_decoder_layers, 
                                       dropout_rate=dropout_rate)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )
        
    def forward(self, observations):
        num_routes = torch.argmax(observations["num_routes"])

        routes = []
        for i in range(num_routes):
            routes.append({
                f"x": observations[f"x_{i}"],
                f"edge_index": observations[f"edge_index_{i}"],
                f"edge_attr": observations[f"edge_attr_{i}"]
            })

        observations = {"x": observations["x"],
                        "edge_index": observations["edge_index"],
                        "edge_attr": observations["edge_attr"]}

        topology_vector = self.topology(observations) #N,L,E
        routes_vectors = [self.route(route) for route in routes] # R x (N, L, E)

        outs = []
        for i in range(len(routes_vectors)):
            if routes_vectors[i].ndim == 2:
                routes_vectors[i] = routes_vectors[i].unsqueeze(0)
            if topology_vector.ndim == 2:
                topology_vector = topology_vector.unsqueeze(0)
            out = self.transformer(topology_vector, routes_vectors[i]) # N,L,E
            out = torch.mean(out, dim=1) # N,E
            outs.append(out)
        
        out = torch.stack(outs, dim=1) # N,R,E
        out = self.ffn(out) #N,R,E
        return out
    
class Actor(nn.Module):
    def __init__(self, feature_dim: int, num_actions: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_actions = num_actions

        self.mlp_a = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
        )

        self.mlp_b = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
        )


    def forward(self, x):
        _, R, _ = x.shape
        actions = []

        for i in range(R):
            actions.append(self.mlp_a(x[:, i, :]))
            actions.append(self.mlp_b(x[:, i, :]))

        actions = torch.cat(actions, dim=1) #N,R*2
        if actions.ndim == 2:
            N, n = actions.shape
            actions = torch.cat([actions, torch.zeros((N,self.num_actions-n), device=x.device)], dim=-1)
        else:
            n = actions.shape[0]
            actions = torch.cat([actions, torch.zeros(self.num_actions-n, device=x.device)], dim=-1)
        return actions
    
class Critic(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

        self.mlp_a = nn.Sequential(
            nn.Linear(feature_dim, feature_dim*2),
            nn.ReLU(),
            nn.Linear(feature_dim*2, feature_dim*4),
        )

        self.mlp_b = nn.Sequential(
            nn.Linear(feature_dim*4, feature_dim*2),
            nn.ReLU(),
            nn.Linear(feature_dim*2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, x):

        a = self.mlp_a(x).mean(dim=1) #N,F
        a = self.mlp_b(a) #N,1

        return a

class GNNPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        activation_fn=nn.ReLU,
        features_extractor_class=None,
        features_extractor_kwargs=None,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            share_features_extractor=True,
            features_extractor_kwargs=features_extractor_kwargs,
            **kwargs,
        )

        feature_dim = self.features_extractor.features_dim

        self.mlp_extractor = nn.Identity()
        self.actor = Actor(feature_dim, action_space.n)
        self.critic = Critic(feature_dim)
        self.value_net = self.critic
        self.action_net = self.actor

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in sum([list(x.children()) for x in self.actor.children()], []) + sum([list(x.children()) for x in self.critic.children()], []):
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        features = self.extract_features(obs)  # shared
        latent_pi = self.actor(features)
        latent_vf = self.critic(features)
        
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, latent_vf, log_prob
    
    def predict(self, observation, state = None, episode_start = None, deterministic = False):
        features = self.extract_features(observation)  # shared
        latent_pi = self.actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions
    
    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        latent_pi = self.actor(features)
        latent_vf = self.value_net(features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return latent_vf, log_prob, entropy

    def _get_action_dist_from_latent(self, latent: torch.Tensor):
        return self.action_dist.proba_distribution(latent)
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(obs)
        return self.value_net(features)
    
    def set_training_mode(self, mode: bool) -> None:
        super().set_training_mode(mode)
        self.features_extractor.train(mode)
        self.actor.train(mode)
        self.critic.train(mode)
        self.value_net.train(mode)
    
    