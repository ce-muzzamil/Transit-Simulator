import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from env import TransitNetworkEnv

class GATv2FeatureExtractor(nn.Module):
    def __init__(self, observation_space, hidden_dim=128, num_heads=4, out_dim=256):
        super().__init__()

        in_channels = observation_space["x"].shape[1]
        edge_dim = observation_space["edge_attr"].shape[-1]

        self.gat1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=True,
            edge_dim=edge_dim 
        )

        self.gat2 = GATv2Conv(
            in_channels=hidden_dim * num_heads,
            out_channels=out_dim,
            heads=1,
            concat=True,
            edge_dim=edge_dim
        )

        self.dropout = nn.Dropout(0.2)

    
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

        x = global_mean_pool(x, batch) #N,E
        
        return x

class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, hidden_dim=128, num_heads=4, out_dim=256, env: TransitNetworkEnv=None):
        super().__init__(observation_space, out_dim)
        self.feature_dim = out_dim
        self.topology = GATv2FeatureExtractor(observation_space, hidden_dim, num_heads, out_dim)
        self.route = GATv2FeatureExtractor(observation_space, hidden_dim, num_heads, out_dim)
        self.fcn = nn.Sequential(
            nn.Linear(2*out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.env = env
        
    def forward(self, observations):
        routes = self.env.get_sub_graphs(observations)

        topology_vector = self.topology(observations) #N,E
        topology_vector = topology_vector.repeat(1, len(routes), 1) # N, R, E
        routes_vectors = torch.stack([self.route(route) for route in routes], 1) # N, R, E

        out = torch.cat([topology_vector, routes_vectors], dim=-1)
        out = self.fcn(out) #N,R,E
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
            actions = torch.cat([actions, torch.zeros((N,self.num_actions-n))], dim=-1)
        else:
            n = actions.shape[0]
            actions = torch.cat([actions, torch.zeros(self.num_actions-n)], dim=-1)
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

    def _get_action_dist_from_latent(self, latent: torch.Tensor):
        return self.action_dist.proba_distribution(latent)