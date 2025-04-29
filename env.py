import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from transit_system import TransitSystem
import torch
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
import time

class TransitNetworkEnv(gym.Env):
    def __init__(self, is_training=True, seed=0):
        np.random.seed(seed)
        self.is_training = is_training
        with open("transit_system_config.json", "r") as file:
            self.transit_system_config = json.load(file)

        self.hours_of_opperation_per_day = self.transit_system_config[
            "hours_of_opperation_per_day"
        ]
        self.analysis_period_sec = self.transit_system_config["analysis_period_sec"]
        self.analysis_period_days = self.transit_system_config["analysis_period_days"]

        self.current_day = 0
        self.current_time = 0

        self.edge_data = {}
        self.nodes_in_routes = {}

        self.alpha_1 = 1
        self.alpha_2 = 1

        self.max_routes = max(self.transit_system_config["max_num_route_per_toplogy"], 10)
        self.max_route_nodes = max(self.transit_system_config["max_num_stops_per_route"], 10)
        self.max_nodes = self.max_routes * self.max_route_nodes
        self.max_route_edges = self.max_route_nodes * 4
        self.max_edges = self.max_nodes * 4
        self.num_node_features = 18
        
        self.max_exit_nodes_per_route = 2

        self.observation_space = spaces.Dict(
            {
                "num_routes": spaces.Discrete(self.max_routes),
                "x": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_nodes, self.num_node_features),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    low=0, high=np.inf, shape=(2, self.max_edges), dtype=np.int64
                ),
                "edge_attr": spaces.Box(
                    low=0, high=np.inf, shape=(self.max_edges, 1), dtype=np.float32
                ),
                **{
                    **{
                        f"x_{i}": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(self.max_route_nodes, self.num_node_features),
                            dtype=np.float32,
                        )
                        for i in range(self.max_routes)
                    },
                    **{
                        f"edge_index_{i}": spaces.Box(
                            low=0,
                            high=np.inf,
                            shape=(2, self.max_route_edges),
                            dtype=np.int64,
                        )
                        for i in range(self.max_routes)
                    },
                    **{
                        f"edge_attr_{i}": spaces.Box(
                            low=0,
                            high=np.inf,
                            shape=(self.max_route_edges, 1),
                            dtype=np.float32,
                        )
                        for i in range(self.max_routes)
                    },
                },
            }
        )

        self.action_space = spaces.MultiBinary(
            self.max_routes * self.max_exit_nodes_per_route
        )
        self.seed = seed
        self.seeds = [self.seed]

    def _reset(self, hard_reset=True):
        if hard_reset:
            if self.is_training:
                self.seed = np.random.randint(
                    0, self.transit_system_config["max_training_seed"]
                )
            else:
                self.seed = np.random.randint(
                    self.transit_system_config["max_training_seed"],
                    self.transit_system_config["max_training_seed"]
                    + self.transit_system_config["max_number_of_testing_seeds"],
                )
            self.del_data()

        self.transit_system = TransitSystem(
            **self.transit_system_config, seed=self.seed
        )
        self.num_nodes = len(self.transit_system.topology.nodes)
        self.num_edges = len(self.transit_system.topology.routes)
        self.num_routes = self.transit_system.topology.num_routes
        self.max_num_buses = 15000000

        if self.num_routes == 0:
            return

        obs = self.get_graph()
        subgraphs = self.get_sub_graphs(obs)
        for subgraph in subgraphs:
            obs.update(subgraph)

        if self.num_routes < self.max_routes:
            for i in range(self.max_routes - self.num_routes):
                sg = {
                    f"x_{self.num_routes + i}": subgraph[f"x_{self.num_routes-1}"],
                    f"edge_index_{self.num_routes + i}": subgraph[
                        f"edge_index_{self.num_routes-1}"
                    ],
                    f"edge_attr_{self.num_routes + i}": subgraph[
                        f"edge_attr_{self.num_routes-1}"
                    ],
                }
                obs.update(sg)

        obs["num_routes"] = self.num_routes

        return obs, {}

    def reset(self, hard_reset=True, *args, **kwargs):
        done = False
        while not done:
            try:
                output = self._reset(hard_reset=hard_reset)
                if hard_reset:
                    if (
                        self.num_routes > 0
                        and self.transit_system is not None
                    ):
                        self.seeds.append(self.seed)
                        done = True
                    else:
                        
                        np.random.seed(int(str(time.time()).split(".")[-1]))
                else:
                    done = True
            except:
                # print("error:", self.seed)
                time.sleep(0.5)
                np.random.seed(int(str(time.time()).split(".")[-1]))
                
        return output

    def get_updated_node_data(self):
        data = []

        buses_data = {}
        for route_id in self.nodes_in_routes.keys():
            buses_data[route_id] = (
                len(
                    [
                        bus
                        for bus in self.transit_system.buses
                        if bus.service_route == route_id and not bus.reversed
                    ]
                ),
                len(
                    [
                        bus
                        for bus in self.transit_system.buses
                        if bus.service_route == route_id and bus.reversed
                    ]
                ),
            )

        for node in self.transit_system.topology.nodes:
            x = node.get_array()
            if node.associated_route != -1:
                x = np.append(x, buses_data[node.associated_route][0])
                x = np.append(x, buses_data[node.associated_route][1])
            else:
                x = np.append(x, -1)
                x = np.append(x, -1)
            x = np.append(x, np.sin(2*np.pi*self.current_time/(self.hours_of_opperation_per_day*3600)))
            data.append(x.astype(np.float32))

        return data

    def get_graph(self):
        graph = self.transit_system.topology.topology.copy()

        routes = self.transit_system.topology.routes
        if len(self.edge_data) == 0:
            for edge in graph.edges():
                for route in routes:
                    if route.node_pair_id == edge or route.node_pair_id[::-1] == edge:
                        self.edge_data[edge] = {"edge_attr": route.distance/1000.0}
                        break

        nx.set_edge_attributes(graph, self.edge_data)

        if len(self.nodes_in_routes) == 0:
            self.nodes_in_routes = {
                k: set()
                for k in sorted(self.transit_system.topology.route_attributes.keys())
            }
            for route in routes:
                self.nodes_in_routes[route.route_id].add(route.node_u.node_id)
                self.nodes_in_routes[route.route_id].add(route.node_v.node_id)
            self.nodes_in_routes = {
                k: list(self.nodes_in_routes[k]) for k in self.nodes_in_routes.keys()
            }

        self.graph: nx.Graph = graph

        data = self.get_updated_node_data()
        obs = from_networkx(graph)
        obs.x = torch.from_numpy(np.stack(data, axis=0))
        self.node_indices = {v: k for k, v in enumerate(self.graph.nodes)}
        return self.fix_obs_shape(obs)

    def del_data(self):
        self.edge_data = {}
        self.nodes_in_routes = {}

    def update_graph(self):
        data = self.get_updated_node_data()
        obs = from_networkx(self.graph)
        obs.x = torch.from_numpy(np.stack(data, axis=0))
        return self.fix_obs_shape(obs)

    def fix_obs_shape(self, obs: Data, is_subgraph=False):
        if is_subgraph:
            max_edges = self.max_route_edges
            max_nodes = self.max_route_nodes
        else:
            max_edges = self.max_edges
            max_nodes = self.max_nodes

        if obs.edge_attr.ndim == 1:
            obs.edge_attr = obs.edge_attr.unsqueeze(1)

        if obs.x.ndim > 2:
            N = obs.x.shape[0]
            x = []
            edge_attr = []
            edge_index = []
            for i in range(N):
                x.append(
                    torch.nn.functional.pad(
                        obs.x[i],
                        (0, 0, 0, max_nodes - obs.x[i].shape[0]),
                        mode="constant",
                        value=0,
                    )
                )
                edge_index.append(
                    torch.nn.functional.pad(
                        obs.edge_index[i],
                        (0, max_edges - obs.edge_index[i].shape[1], 0, 0),
                        mode="constant",
                        value=0,
                    )
                )
                edge_attr.append(
                    torch.nn.functional.pad(
                        obs.edge_attr[i],
                        (0, 0, 0, max_edges - obs.edge_attr[i].shape[0]),
                        mode="constant",
                        value=0,
                    )
                )

            obs.x = torch.stack(x, dim=0)
            edge_index = torch.stack(edge_index, dim=0)
            edge_attr = torch.stack(edge_attr, dim=0)
        else:
            obs.x = torch.nn.functional.pad(
                obs.x,
                (0, 0, 0, max_nodes - obs.x.shape[0]),
                mode="constant",
                value=0,
            )
            obs.edge_index = torch.nn.functional.pad(
                obs.edge_index,
                (0, max_edges - obs.edge_index.shape[1], 0, 0),
                mode="constant",
                value=0,
            )
            obs.edge_attr = torch.nn.functional.pad(
                obs.edge_attr,
                (0, 0, 0, max_edges - obs.edge_attr.shape[0]),
                mode="constant",
                value=0,
            )

        return {"x": obs.x, "edge_index": obs.edge_index, "edge_attr": obs.edge_attr}

    def step(self, action: np.ndarray):
        """
        Arguments:
        ---------
        `action`: contains the array of actions for each route id. Each array of the action consists of a list of two binary variables.
        these binary variables corresponds to each exit node of the route and the binary variable is to indicate whether to add a bus for that exit node or not.
        Since, a single model is used for all routes, The len of action can be changed from toplogy to toplogy but the mechanism will not fail.
        """
        action = action[: self.num_routes * 2]

        for i, decision in enumerate(action):
            if decision == 1:
                self.transit_system.add_bus_on_route(
                    i // 2, reversed=False if i % 2 == 0 else True
                )

        reward, reward_info = self.reward(action)
        obs: dict = self.update_graph()

        truncated = False
        terminated = False

        self.transit_system.step(self.current_time)

        if (
            self.current_time + self.analysis_period_sec
        ) >= self.hours_of_opperation_per_day * 3600:
            self.current_day += 1
            self.current_time = 0
            obs, _ = self.reset(hard_reset=False)
        else:
            self.current_time = self.current_time + self.analysis_period_sec

        if self.current_day >= self.analysis_period_days:
            truncated = True

        info = {**reward_info}

        if len(self.transit_system.buses) > self.max_num_buses:
            terminated = True
            reward = -100

        subgraphs = self.get_sub_graphs(obs)
        for subgraph in subgraphs:
            obs.update(subgraph)

        if self.num_routes < self.max_routes:
            for i in range(self.max_routes - self.num_routes):
                sg = {
                    f"x_{self.num_routes + i}": subgraph[f"x_{self.num_routes-1}"],
                    f"edge_index_{self.num_routes + i}": subgraph[
                        f"edge_index_{self.num_routes-1}"
                    ],
                    f"edge_attr_{self.num_routes + i}": subgraph[
                        f"edge_attr_{self.num_routes-1}"
                    ],
                }
                obs.update(sg)

        obs["num_routes"] = self.num_routes
        return obs, reward, terminated, truncated, info

    def get_sub_graphs(self, obs: dict) -> list[Data]:
        if obs["edge_index"].ndim == 2:
            subgraphs: list[dict] = []
            for route_id in self.nodes_in_routes.keys():
                indices = torch.tensor(
                    [self.node_indices[i] for i in self.nodes_in_routes[route_id]]
                ).long()
                edge_index, edge_attr = subgraph(
                    indices,
                    obs["edge_index"].long(),
                    obs["edge_attr"],
                    relabel_nodes=True,
                    num_nodes=self.num_nodes,
                )
                sub_data = Data(
                    x=obs["x"][indices],
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=None if "y" not in obs else obs["y"][indices],
                )
                subgraphs.append(self.fix_obs_shape(sub_data, is_subgraph=True))
        else:
            subgraphs: list[dict] = []
            for route_id in self.nodes_in_routes.keys():
                indices = torch.tensor(
                    [self.node_indices[i] for i in self.nodes_in_routes[route_id]]
                ).long()
                batch_size = obs["edge_index"].shape[0]
                sub_datas = []
                for i in range(batch_size):
                    edge_index, edge_attr = subgraph(
                        indices,
                        obs["edge_index"][i].long(),
                        obs["edge_attr"][i],
                        relabel_nodes=True,
                        num_nodes=self.num_nodes,
                    )
                    sub_data = Data(
                        x=obs["x"][i][indices],
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=None if "y" not in obs else obs["y"][i][indices],
                    )
                    sub_datas.append(self.fix_obs_shape(sub_data, is_subgraph=True))

                sub_data = Data(
                    x=torch.stack([sub_data["x"] for sub_data in sub_datas], dim=0),
                    edge_index=torch.stack(
                        [sub_data["edge_index"] for sub_data in sub_datas], dim=0
                    ),
                    edge_attr=torch.stack(
                        [sub_data["edge_attr"] for sub_data in sub_datas], dim=0
                    ),
                    y=(
                        None
                        if "y" not in obs
                        else torch.stack(
                            [sub_data["y"] for sub_data in sub_datas], dim=0
                        )
                    ),
                )

                subgraphs.append(sub_data.to_dict())

        return [
            {k + f"_{i}": v for k, v in subgraph.items()}
            for i, subgraph in enumerate(subgraphs)
        ]

    def reward(self, actions) -> float:

        sum_reward_1 = 0
        sum_reward_2 = 0
        sum_reward_3 = 0

        for i, action in enumerate(actions):
            route_id = int(i // 2)
            is_reversed = not (i % 2 == 0)

            reward_1 = 0
            node_counts = 0
            num_passengers = []

            for node in self.transit_system.topology.nodes:
                if route_id in node.affliated_route_ids:
                    num_passengers.append(len(node.passengers))
                    demand = len(node.passengers)/2 + 1 # devided by 2 to get approx only one side (reversed or not)
                    capacity = 1
                    for bus in self.transit_system.buses:
                        if bus.service_route == route_id and bus.reversed == is_reversed:
                            if node in bus.to_go:
                                capacity += bus.capacity - len(bus.passengers)

                    demand_capacity_ratio = demand / max(capacity, 1e-5)
                    node_counts += 1

                    if (demand_capacity_ratio < 1 and action == 1) or (
                        demand_capacity_ratio > 1 and action == 0
                    ):
                        reward_1 += -2

            sum_reward_1 += reward_1 / node_counts if node_counts > 0 else 0   

            avg_waiting_time = [
                np.max([passenger.waiting_time for passenger in node.passengers])
                for node, num_passenger in zip(
                    [
                        node
                        for node in self.transit_system.topology.nodes
                        if route_id in node.affliated_route_ids
                    ],
                    num_passengers,
                )
                if num_passenger > 0
            ]

            avg_stranding_count = [
                np.max([passenger.stranding_counts for passenger in node.passengers])
                for node, num_passenger in zip(
                    [
                        node
                        for node in self.transit_system.topology.nodes
                        if route_id in node.affliated_route_ids
                    ],
                    num_passengers,
                )
                if num_passenger > 0
            ]

            if len(avg_waiting_time) > 0:
                avg_waiting_time = np.mean(avg_waiting_time)//15.  # minutes
            else:
                avg_waiting_time = 0.0

            if len(avg_stranding_count) > 0:
                avg_stranding_count = np.mean(avg_stranding_count)
            else:
                avg_stranding_count = 0  # counts

            if avg_waiting_time == 0:
                pass
            else:
                sum_reward_2 += -avg_waiting_time/60
            

            if avg_stranding_count > 0:
                sum_reward_2 += -2

            if action == 1:
                expence_of_bus_journey = 7  # 1.5 km/leter
            else:
                expence_of_bus_journey = 0

            sum_reward_3 += -expence_of_bus_journey

        sum_reward_1 = sum_reward_1 / len(actions)
        sum_reward_2 = sum_reward_2 / len(actions)
        sum_reward_3 = sum_reward_3 / len(actions)

        reward = sum_reward_1 + sum_reward_2 + sum_reward_3

        reward_info = {
            "reward_type_1": sum_reward_1,
            "reward_type_2": sum_reward_2,
            "reward_type_3": sum_reward_3,
            "reward": reward,
        }
        return reward, reward_info

    def render(self):
        pass
