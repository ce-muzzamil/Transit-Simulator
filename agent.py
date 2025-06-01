import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.distributions import Categorical
from copy import deepcopy
import itertools



def to_torch(obs_):
    obs = deepcopy(obs_)
    for k1 in obs:
        for k2 in obs[k1]:
            obs[k1][k2] = torch.from_numpy(obs[k1][k2]).to(torch.float32)
    return obs


def to_device(obs, device="cpu"):
    if isinstance(obs, dict):
        for k1 in obs:
            if isinstance(obs[k1], dict):
                for k2 in obs[k1]:
                    obs[k1][k2] = obs[k1][k2].to(device)
            else:
                obs[k1] = obs[k1].to(device)
    else:
        obs = obs.to(device)
    return obs


def detach_grads(obs):
    if isinstance(obs, dict):
        for k1 in obs:
            if isinstance(obs[k1], dict):
                for k2 in obs[k1]:
                    obs[k1][k2] = obs[k1][k2].detach()
            else:
                obs[k1] = obs[k1].detach()
    else:
        obs = obs.detach()
    return obs


def batch_obs(obs: list):
    bobs = {}
    keys = obs[0].keys()
    for k in keys:
        bobs[k] = torch.stack([obs[i][k] for i in range(len(obs))], dim=0)
    return bobs


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


# class FeatureExtractor(nn.Module):
#     def __init__(
#         self,
#         observation_space,
#         gnn_hidden_dim=128,
#         gnn_num_heads=4,
#         embed_size=256,
#         transformer_num_heads=4,
#         num_encoder_layers=6,
#         num_decoder_layers=6,
#         dropout_rate=0.0,
#     ):
#         super().__init__()

#         self.feature_dim = embed_size
#         self.topology = GATv2FeatureExtractor(
#             observation_space["x"].shape[-1],
#             observation_space["edge_attr"].shape[-1],
#             gnn_hidden_dim,
#             gnn_num_heads,
#             embed_size,
#             dropout_rate=dropout_rate,
#         )

#         self.route = GATv2FeatureExtractor(
#             observation_space["x_route"].shape[-1],
#             observation_space["edge_attr_route"].shape[-1],
#             gnn_hidden_dim,
#             gnn_num_heads,
#             embed_size,
#             dropout_rate=dropout_rate,
#         )

#         self.transformer = Transformer(
#             embed_size=embed_size,
#             num_heads=transformer_num_heads,
#             num_encoder_layers=num_encoder_layers,
#             num_decoder_layers=num_decoder_layers,
#             dropout_rate=dropout_rate,
#         )

#     def forward(self, observations):
#         route = {
#             f"x": observations[f"x_route"],
#             f"edge_index": observations[f"edge_index_route"],
#             f"edge_attr": observations[f"edge_attr_route"],
#         }

#         observations = {
#             "x": observations["x"],
#             "edge_index": observations["edge_index"],
#             "edge_attr": observations["edge_attr"],
#         }

#         topology_vector = self.topology(observations)  # N,L,E
#         routes_vector = self.route(route)  # (N, L, E)

#         if routes_vector.ndim == 2:
#             routes_vector = routes_vector.unsqueeze(0)

#         if topology_vector.ndim == 2:
#             topology_vector = topology_vector.unsqueeze(0)

#         out = self.transformer(topology_vector, routes_vector)  # N,L,E
#         out = torch.mean(out, dim=1)  # N,E
#         return out

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
        self.route = GATv2FeatureExtractor(
            observation_space["x_route"].shape[-1],
            observation_space["edge_attr_route"].shape[-1],
            gnn_hidden_dim,
            gnn_num_heads,
            embed_size,
            dropout_rate=dropout_rate,
        )

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(embed_size, transformer_num_heads, dropout_rate)
                for _ in range(num_encoder_layers)
            ]
        )

    def forward(self, observations):
        route = {
            f"x": observations[f"x_route"],
            f"edge_index": observations[f"edge_index_route"],
            f"edge_attr": observations[f"edge_attr_route"],
        }

        routes_vector = self.route(route)  # (N, L, E)

        if routes_vector.ndim == 2:
            routes_vector = routes_vector.unsqueeze(0)

        for layer in self.encoder_layers:
            routes_vector = layer(routes_vector)

        out = torch.mean(routes_vector, dim=1)  # N,E
        return out

class Model(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        gnn_hidden_dim=128,
        gnn_num_heads=4,
        embed_size=256,
        transformer_num_heads=4,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout_rate=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.num_actions = action_space.n

        self.feature_extractor = FeatureExtractor(
            observation_space=observation_space,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_num_heads=gnn_num_heads,
            embed_size=embed_size,
            transformer_num_heads=transformer_num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout_rate=dropout_rate,
        )

        self.actor = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, self.num_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, 1),
        )

    def forward(self, x):
        embed = self.feature_extractor(x)
        logits = self.actor(embed).squeeze(-1)
        value = self.critic(embed).squeeze(-1)
        return logits, value

def collect_rollout(env, model, rollout_len=1080, device="cpu", hard_reset=True):
    obs, _ = env.reset(hard_reset=hard_reset)
    (
        obs_buf,
        action_buf,
        reward_buf,
        terminated_buf,
        truncated_buf,
        info_buf,
        logp_buf,
        value_buf,
    ) = (
        {agent_id: [] for agent_id in env.possible_agents},
        {agent_id: [] for agent_id in env.possible_agents},
        {agent_id: [] for agent_id in env.possible_agents},
        {agent_id: [] for agent_id in env.possible_agents},
        {agent_id: [] for agent_id in env.possible_agents},
        [],
        {agent_id: [] for agent_id in env.possible_agents},
        {agent_id: [] for agent_id in env.possible_agents},
    )

    killed_agents = set()
    for step_count in range(rollout_len):
        obs = to_torch(obs)

        actions = {}
        for index, agent_id in enumerate(env.possible_agents):
            if agent_id in killed_agents:
                actions[agent_id] = 0
                continue

            with torch.no_grad():
                logits, value = model(to_device(obs[agent_id], device=device))
                probs = F.softmax(logits, dim=-1)

            dist = Categorical(probs=probs)
            action = dist.sample()

            obs_buf[agent_id].append(to_device(detach_grads(obs[agent_id]), device="cpu"))
            action_buf[agent_id].append(action.item())
            logp_buf[agent_id].append(dist.log_prob(action).detach().cpu())
            value_buf[agent_id].append(value.squeeze(-1).detach().cpu())

            actions[agent_id] = action.item()

        next_obs, reward, terminated, truncated, info = env.step(actions)
        for agent_id in env.possible_agents:
            if agent_id not in killed_agents:                
                reward_buf[agent_id].append(torch.tensor(reward[agent_id], dtype=torch.float32))
                terminated_buf[agent_id].append(terminated[agent_id])
                truncated_buf[agent_id].append(truncated[agent_id])

                if terminated[agent_id] or truncated[agent_id]:
                    if agent_id not in killed_agents:
                        killed_agents.add(agent_id)
                if terminated[agent_id]:
                    print(f"Agent {agent_id} terminated at step {step_count}.")
                
        info_buf.append(info)
        
        obs = next_obs
        if len(killed_agents) == len(env.possible_agents):
            break

    return (
        obs_buf,
        action_buf,
        reward_buf,
        terminated_buf,
        truncated_buf,
        info_buf,
        logp_buf,
        value_buf,
    )


def ppo_update(
    model,
    optimizer,
    obs_buf,
    action_buf,
    reward_buf,
    terminated_buf,
    truncated_buf,
    logp_buf,
    value_buf,
    gamma=0.995,
    lam=0.95,
    clip_ratio=0.2,
    entropy_coef=0.01,
    vf_coef=1.0,
    target_kl=0.015,
    epochs=5,
    batch_size=32,
    device="cpu",
):
    policy_losses, value_losses = [], []

    for agent_id in obs_buf.keys():
        done_buf = [t or tr for t, tr in zip(terminated_buf[agent_id], truncated_buf[agent_id])]
        T = len(reward_buf[agent_id])

        # Bootstrap value
        if T > 0:
            if done_buf[-1]:
                last_value = 0.0
            else:
                with torch.no_grad():
                    final_obs = to_device(obs_buf[agent_id][-1], device=device)
                    if final_obs.dim() == 1:
                        final_obs = final_obs.unsqueeze(0)
                    _, last_value_tensor = model(final_obs)
                    last_value = last_value_tensor.squeeze(-1).item()
        else:
            last_value = 0.0

        # Compute GAE and returns
        returns, advs = [], []
        gae = 0.0
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else value_buf[agent_id][t + 1]
            next_non_terminal = 1.0 - float(done_buf[t])
            delta = reward_buf[agent_id][t] + gamma * next_value * next_non_terminal - value_buf[agent_id][t]
            gae = delta + gamma * lam * next_non_terminal * gae
            advs.insert(0, gae)
            returns.insert(0, gae + value_buf[agent_id][t])

        advs = torch.tensor(advs, dtype=torch.float32, device=device)
        advs = (advs-advs.min()) / (advs.max()-advs.min() + 1e-8)  # Normalize advantages
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        old_logps = torch.stack(logp_buf[agent_id]).to(device)
        
        indices = torch.randperm(T)

        for epoch in range(epochs):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy = 0.0
            num_batches = 0

            for i in range(0, T, batch_size):
                batch_indices = indices[i:i + batch_size]

                obs_batch_list = [obs_buf[agent_id][idx] for idx in batch_indices]
                obs_batch = batch_obs(obs_batch_list)
                obs_batch = to_device(obs_batch, device=device)

                logits, new_values = model(obs_batch)
                dist = Categorical(logits=logits)
                entropy = dist.entropy().mean()

                actions = torch.tensor([action_buf[agent_id][idx] for idx in batch_indices], device=device)
                new_logp = dist.log_prob(actions)
                old_logp = old_logps[batch_indices]

                adv_batch = advs[batch_indices]
                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                ret_batch = returns[batch_indices]
                v_pred = new_values.squeeze(-1)
                value_loss = F.mse_loss(v_pred, ret_batch)

                optimizer.zero_grad()
                (policy_loss - entropy_coef * entropy + value_loss * vf_coef).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()


                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy += entropy.item()
                num_batches += 1

            # ------------- KL Divergence Check -------------
            if num_batches > 0:
                with torch.no_grad():
                    sample_size = min(64, T)
                    sample_indices = torch.randperm(T)[:sample_size]
                    sample_obs = batch_obs([obs_buf[agent_id][idx] for idx in sample_indices])
                    sample_obs = to_device(sample_obs, device=device)
                    sample_actions = torch.tensor([action_buf[agent_id][idx] for idx in sample_indices], device=device)

                    new_logits, _ = model(sample_obs)
                    new_dist = Categorical(logits=new_logits)
                    new_logp = new_dist.log_prob(sample_actions)
                    old_logp_sample = old_logps[sample_indices]
                    logp_diff = new_logp - old_logp_sample
                    approx_kl = ((torch.exp(logp_diff) - 1) - logp_diff).mean()
                    if approx_kl > target_kl:
                        break

            policy_losses.append(epoch_policy_loss / num_batches)
            value_losses.append(epoch_value_loss / num_batches)

    if len(policy_losses) == 0:
        return 0.0, 0.0
    return np.mean(policy_losses), np.mean(value_losses)
