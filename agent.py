import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.distributions import Categorical
from copy import deepcopy

from logger import TrainingEpisodeLogger

def to_torch(obs_):
    obs = deepcopy(obs_)
    for k1 in obs:
        for k2 in obs[k1]:
            if not isinstance(obs[k1][k2], torch.Tensor):
                obs[k1][k2] = torch.from_numpy(obs[k1][k2]).to(torch.float32)
            else:
                obs[k1][k2] = obs[k1][k2].to(torch.float32)
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

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.gat1 = GATv2Conv(
            in_channels=hidden_dim,
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
            assert N == 1 # added after transitioning to no-batch intervention
            outs = []
            for i in range(N):
                outs.append(gat(x[i], edge_index[i], edge_attr[i]))
            return torch.stack(outs, dim=0)

    def forward(self, data):
        x, edge_index, edge_attr = (
            data["x"],
            data["edge_index"].long(),
            data["edge_attr"],
        )

        if torch.isnan(x).any():
            print("Found NaNs in obs.x")
        if torch.isnan(edge_attr).any():
            print("Found NaNs in obs.edge_attr")
        if torch.isnan(edge_index).any():
            print("Found NaNs in obs.edge_index")
            
        x = self.mlp(x)
        x = self.process_for_gat(self.gat1, x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.process_for_gat(self.gat2, x, edge_index, edge_attr)
        x = F.relu(x)
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
        attn_output, _ = self.mha(x, x, x)
        x = self.norm_1(x + self.dropout_1(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm_2(x + self.dropout_2(ffn_output))

        return x


class DecoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_rate=0.0):
        super().__init__()

        self.self_mha = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.norm_1 = nn.LayerNorm(embed_size)
        self.dropout_1 = nn.Dropout(dropout_rate)

        self.cross_mha = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.norm_2 = nn.LayerNorm(embed_size)
        self.dropout_2 = nn.Dropout(dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )

        self.norm_3 = nn.LayerNorm(embed_size)
        self.dropout_3 = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output):
        self_attn_output, _ = self.self_mha(x, x, x)
        x = self.norm_1(x + self.dropout_1(self_attn_output))

        cross_attn_output, _ = self.cross_mha(x, enc_output, enc_output)
        x = self.norm_2(x + self.dropout_2(cross_attn_output))

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

        self.critic_delayed = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, 1),
        )

        self.critic_immediate = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, 1),
        )

    def forward(self, x):
        embed = self.feature_extractor(x)
        logits = self.actor(embed).squeeze(-1)
        value_immediate, value_delayed = self.critic_immediate(embed).squeeze(
            -1
        ), self.critic_delayed(embed).squeeze(-1)
        return logits, value_immediate, value_delayed


def fixed_policy(interval):
    def wrapped(time_step):
        if time_step % interval == 0:
            return 1
        else:
            return 0
    return wrapped


def collect_rollout(
    env, model, rollout_len=1080, device="cpu", hard_reset=True, testing=False
):
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
        {agent_id: [] for agent_id in env.possible_agents},
        {agent_id: [] for agent_id in env.possible_agents},
        {agent_id: [] for agent_id in env.possible_agents},
    )

    killed_agents = set()
    sc = 0
    num_killed = 0
    for step_count in range(rollout_len):
        obs = to_torch(obs)

        actions = {}
        for index, agent_id in enumerate(env.possible_agents):
            if agent_id in killed_agents:
                actions[agent_id] = 0
                continue

            if isinstance(model, nn.Module):
                with torch.no_grad():
                    logits, value_imm, value_del = model(
                        to_device(obs[agent_id], device=device)
                    )
                    value_buf[agent_id].append(
                        (
                            value_imm.squeeze(-1).detach().cpu(),
                            value_del.squeeze(-1).detach().cpu(),
                        )
                    )

                if not testing:
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    logp_buf[agent_id].append(dist.log_prob(action).detach().cpu())
                    action = action.item()

                else:
                    # action = torch.argmax(logits, dim=-1)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
            else:
                if model == "random":
                    action = np.random.choice((0, 1), [])
                else:
                    action = model(env.current_time)

            obs_buf[agent_id].append(
                to_device(detach_grads(obs[agent_id]), device="cpu")
            )
            action_buf[agent_id].append(action)

            actions[agent_id] = action

        next_obs, reward, terminated, truncated, info = env.step(actions)
        for agent_id in env.possible_agents:
            if agent_id not in killed_agents:
                reward_buf[agent_id].append(
                    torch.tensor(reward[agent_id], dtype=torch.float32)
                )
                info_buf[agent_id].append(info[agent_id])
                terminated_buf[agent_id].append(terminated[agent_id])
                truncated_buf[agent_id].append(truncated[agent_id])

                if terminated[agent_id] or truncated[agent_id]:
                    if agent_id not in killed_agents:
                        killed_agents.add(agent_id)
                if terminated[agent_id]:
                    num_killed += 1
                    sc += step_count

        obs = next_obs
        if len(killed_agents) == len(env.possible_agents):
            break

    good_buses = 0
    for agent_id in env.possible_agents:
        T = len(reward_buf[agent_id])
        for t in reversed(range(T)):
            current_time = info_buf[agent_id][t]["current_time"]
            additional_reward = None
            for i in range(t, T):
                retired_buses = info_buf[agent_id][i]["retired_buses"]
                for bus in retired_buses:
                    if bus.created_at == current_time:
                        if bus.num_passengers_served / bus.capacity > 0.90:
                            additional_reward = 4
                        elif bus.num_passengers_served / bus.capacity > 0.50:
                            additional_reward = 2
                        elif bus.num_passengers_served / bus.capacity > 0.10:
                            additional_reward = 0
                        elif bus.num_passengers_served / bus.capacity > 0.0:
                            additional_reward = -2
                        else:
                            additional_reward = -4
                        break
                    
                if additional_reward is not None:
                    reward_buf[agent_id][t] += additional_reward
                    info_buf[agent_id][t]["reward_type_3"] += additional_reward
                    good_buses += 1
                    break

    if num_killed > 0:
        print(
            f"Killed {num_killed}/{len(env.possible_agents)} agents at step {int(sc/num_killed)}."
        )
    
    # mean_of_action_0 = []
    # mean_of_action_1 = []
    # for agent_id in env.possible_agents:
    #     agent_actions = action_buf[agent_id]
    #     agent_rewards = reward_buf[agent_id]
    #     for r, a in zip(agent_rewards, agent_actions):
    #         if a == 0:
    #             mean_of_action_0.append(r)
    #         elif a == 1:
    #             mean_of_action_1.append(r)

    # if not testing:
    #     mean_of_action_0 = sum(mean_of_action_0) / len(mean_of_action_0) if mean_of_action_0 else 0.0
    #     mean_of_action_1 = sum(mean_of_action_1) / len(mean_of_action_1) if mean_of_action_1 else 0.0
    #     print(f"Mean of action 0: {mean_of_action_0:.2f}, Mean of action 1: {mean_of_action_1 - mean_of_action_0:.2f}")

    if not testing:
        mean_of_action_0 = np.mean([info_buf[agent_id][t]["reward_type_2"] for agent_id in env.possible_agents for t in range(len(reward_buf[agent_id]))])
        mean_of_action_1 = np.mean([info_buf[agent_id][t]["reward_type_3"] for agent_id in env.possible_agents for t in range(len(reward_buf[agent_id]))])
        print(f"Mean of action 0: {mean_of_action_0:.2f}, Mean of action 1: {mean_of_action_1 - mean_of_action_0:.2f}")

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
    info_buf,
    logp_buf,
    value_buf,
    gamma_imm=0.100,
    gamma_del=0.995,
    lam=0.95,
    clip_ratio=0.2,
    entropy_coef=0.01,
    vf_coef=1.0,
    target_kl=0.015,
    epochs=5,
    batch_size=32,
    device="cpu",
    env=None,
    logger: TrainingEpisodeLogger=None
):
    policy_losses, value_losses, value_losses_imm, value_losses_del, entr = [], [], [], [], []

    for agent_id in obs_buf.keys():
        done_buf = [
            t or tr for t, tr in zip(terminated_buf[agent_id], truncated_buf[agent_id])
        ]
        T = len(reward_buf[agent_id])
        returns_imm, advs_imm = [], []
        returns_del, advs_del = [], []
        gae_imm = 0.0
        gae_del = 0.0
        for t in reversed(range(T)):
            next_non_terminal = 1.0 - float(done_buf[t])
            _gamma_imm = 0.0 if info_buf[agent_id][t]["reward_type_3"] < 0 or action_buf[agent_id][t] == 1 else gamma_imm

            next_value_imm = 0.0 if t == T - 1 else value_buf[agent_id][t + 1][0]
            delta_imm = (
                info_buf[agent_id][t]["reward_type_3"]
                + _gamma_imm * next_value_imm * next_non_terminal
                - value_buf[agent_id][t][0]
            )

            gae_imm = delta_imm + _gamma_imm * lam * next_non_terminal * gae_imm
            advs_imm.insert(0, gae_imm)
            returns_imm.insert(0, gae_imm + value_buf[agent_id][t][0])

            next_value_del = 0.0 if t == T - 1 else value_buf[agent_id][t + 1][1]

            if action_buf[agent_id][t] == 0:
                _gamma_del = gamma_del if info_buf[agent_id][t]["reward_type_2"] == 0 else 0.0
                delta_del = (
                    info_buf[agent_id][t]["reward_type_2"]
                    + _gamma_del * next_value_del * next_non_terminal
                    - value_buf[agent_id][t][1]
                )
                gae_del = delta_del + _gamma_del * lam * next_non_terminal * gae_del
                advs_del.insert(0, gae_del)
                returns_del.insert(0, gae_del + value_buf[agent_id][t][1])
            else:
                delta_del = 0 - value_buf[agent_id][t][1]
                advs_del.insert(0, delta_del)
                returns_del.insert(0, 0)

        advs_imm, advs_del = (
            torch.tensor(advs_imm, dtype=torch.float32, device=device),
            torch.tensor(advs_del, dtype=torch.float32, device=device),
        )

        advs = advs_imm + advs_del
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        returns_imm, returns_del = (
            torch.tensor(returns_imm, dtype=torch.float32, device=device),
            torch.tensor(returns_del, dtype=torch.float32, device=device),
        )

        old_logps = torch.stack(logp_buf[agent_id]).to(device)

        indices = torch.randperm(T)
        for epoch in range(epochs):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_value_loss_imm = 0.0
            epoch_value_loss_del = 0.0
            epoch_entropy = 0.0
            num_batches = 0

            for i in range(0, T, batch_size):
                batch_indices = indices[i : i + batch_size]

                obs_batch_list = [obs_buf[agent_id][idx] for idx in batch_indices]
                logits, new_values_imm, new_values_del  = [], [], []
                for obs in obs_batch_list:
                    _logits, _new_values_imm, _new_values_del = model(to_device(obs, device=device))
                    logits.append(_logits)
                    new_values_imm.append(_new_values_imm)
                    new_values_del.append(_new_values_del)

                logits = torch.stack(logits, dim=0).to(device)
                new_values_imm = torch.stack(new_values_imm, dim=0).to(device)
                new_values_del = torch.stack(new_values_del, dim=0).to(device)

                dist = Categorical(logits=logits)
                entropy = dist.entropy().mean()

                actions = torch.tensor(
                    [action_buf[agent_id][idx] for idx in batch_indices], device=device
                )
                new_logp = dist.log_prob(actions)
                old_logp = old_logps[batch_indices]

                adv_batch = advs[batch_indices]
                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_batch
                
                policy_loss = -torch.min(surr1, surr2).mean()

                ret_batch_imm = returns_imm[batch_indices]
                ret_batch_del = returns_del[batch_indices]

                v_pred_imm = new_values_imm.squeeze(-1)
                value_loss_imm = F.mse_loss(v_pred_imm, ret_batch_imm)

                v_pred_del = new_values_del.squeeze(-1)
                value_loss_del = F.mse_loss(
                    v_pred_del.squeeze(), ret_batch_del.squeeze()
                )
                value_loss = value_loss_imm + value_loss_del

                optimizer.zero_grad()
                (policy_loss - entropy_coef * entropy + value_loss * vf_coef).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_value_loss_imm += value_loss_imm.item()
                epoch_value_loss_del += value_loss_del.item()
                epoch_entropy += entropy.item()
                num_batches += 1

            # ------------- KL Divergence Check -------------
            # if num_batches > 0:
            #     with torch.no_grad():
            #         sample_size = min(64, T)
            #         sample_indices = torch.randperm(T)[:sample_size]
            #         sample_obs = batch_obs(
            #             [obs_buf[agent_id][idx] for idx in sample_indices]
            #         )
            #         sample_obs = to_device(sample_obs, device=device)
            #         sample_actions = torch.tensor(
            #             [action_buf[agent_id][idx] for idx in sample_indices],
            #             device=device,
            #         )

            #         new_logits, _, _ = model(sample_obs)
            #         new_dist = Categorical(logits=new_logits)
            #         new_logp = new_dist.log_prob(sample_actions)
            #         old_logp_sample = old_logps[sample_indices]
            #         logp_diff = new_logp - old_logp_sample
            #         approx_kl = ((torch.exp(logp_diff) - 1) - logp_diff).mean()
            #         if approx_kl > target_kl:
            #             break

            policy_losses.append(epoch_policy_loss / num_batches)
            value_losses.append(epoch_value_loss / num_batches)
            value_losses_imm.append(epoch_value_loss_imm / num_batches)
            value_losses_del.append(epoch_value_loss_del / num_batches)
            entr.append(epoch_entropy / num_batches)

    if logger is not None:
        logger.add_to_pool(
            seed=env.seed,
            delayed_reward=np.mean([info_buf[agent_id][t]["reward_type_2"] for agent_id in env.possible_agents for t in range(len(reward_buf[agent_id]))]),
            immediate_reward=np.mean([info_buf[agent_id][t]["reward_type_3"] for agent_id in env.possible_agents for t in range(len(reward_buf[agent_id]))]),
            policy_loss=np.mean(policy_losses),
            delayed_value_loss=np.mean(value_losses_del),
            immediate_value_loss=np.mean(value_losses_imm),
            entropy=np.mean(entr)
        )
        logger.commit()

    if len(policy_losses) == 0:
        return 0.0, 0.0
    return np.mean(policy_losses), np.mean(value_losses)
