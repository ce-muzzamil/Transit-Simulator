import json
import numpy as np
import pandas as pd
from gymnasium import spaces
from transit_system import TransitSystem
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data
import time


class TransitNetworkEnv:
    def __init__(self, config=None):

        is_training = config["is_training"]
        seed = np.random.seed(config["seed"])

        if "force_seed" in config:
            self.force_seed = config["force_seed"]
        else:
            self.force_seed = None

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

        self.max_routes = max(
            self.transit_system_config["max_num_route_per_toplogy"], 10
        )
        self.max_route_nodes = max(
            self.transit_system_config["max_num_stops_per_route"], 10
        )
        self.max_nodes = self.max_routes * self.max_route_nodes
        self.max_route_edges = self.max_route_nodes * 2
        self.max_edges = self.max_nodes * 4
        self.num_node_features = 26
        self.max_exit_nodes_per_route = 2

        self.agents = [f"agent_{i}" for i in range(self.max_routes * 2)]

        self.possible_agents = [f"agent_{i}" for i in range(self.max_routes * 2)]

        self.rd_2_agent_id = [
            (route_id, reversed)
            for route_id in range(self.max_routes)
            for reversed in [False, True]
        ]

        self.rd_2_agent_id = {k: v for k, v in zip(self.rd_2_agent_id, self.agents)}

        get_action_space = lambda: spaces.Discrete(2)
        get_obs_space = lambda: spaces.Dict(
            {
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
                f"x_route": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_route_nodes, self.num_node_features),
                    dtype=np.float32,
                ),
                f"edge_index_route": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(2, self.max_route_edges),
                    dtype=np.int64,
                ),
                f"edge_attr_route": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(self.max_route_edges, 1),
                    dtype=np.float32,
                ),
            }
        )

        self.observation_spaces = {key: get_obs_space() for key in self.agents}
        self.action_spaces = {key: get_action_space() for key in self.agents}
        self.seed = seed
        self.seeds = [self.seed]

    def _reset(self, hard_reset=True):

        if hard_reset:
            if self.force_seed is not None:
                self.seed = self.force_seed
                self.force_seed = None
            else:
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

        self.avg_waiting_time = {k:0 for k in self.possible_agents}
        self.transit_system = TransitSystem(
            **self.transit_system_config, seed=self.seed
        )

        self.num_nodes = len(self.transit_system.topology.nodes)
        self.num_edges = len(self.transit_system.topology.routes)
        self.num_routes = self.transit_system.topology.num_routes

        self.max_num_buses = 15000000

        if self.num_routes == 0:
            return

    def reset(self, hard_reset=True, *args, **kwargs):
        done = False
        while not done:
            try:
                self._reset(hard_reset=hard_reset)
                if hard_reset:
                    if self.num_routes > 0 and self.transit_system is not None:
                        self.seeds.append(self.seed)
                        done = True
                    else:
                        np.random.seed(int(str(time.time()).split(".")[-1]))
                else:
                    done = True
            except:
                np.random.seed(int(str(time.time()).split(".")[-1]))

        self.directed_sub_routes = {}
        for i in range(self.num_routes):
            for reversed in [False, True]:
                self.transit_system.add_bus_on_route(i, reversed=reversed)
                bus = self.transit_system.buses[0]
                nodes = np.array([node.node_id for node in bus.to_go])
                indices = [j for j in range(len(nodes))]
                edge_index = np.array(list(zip(indices[:-1], indices[1:]))).T
                edge_attrs = np.array(bus.distances[1:]) / 1000.0
                self.transit_system.buses.remove(bus)
                self.transit_system.num_busses_added = 0
                self.directed_sub_routes[(i, reversed)] = (
                    nodes,
                    edge_index,
                    edge_attrs,
                )

        obs = self.get_graph()
        subgraphs = self.get_sub_graphs(obs)
        all_obs = {}

        for key, subgraph in subgraphs.items():
            sub_obs = {**obs, **{k + "_route": v for k, v in subgraph.items()}}
            all_obs[self.rd_2_agent_id[key]] = sub_obs

        for agent_id in self.possible_agents:
            if agent_id not in all_obs:
                all_obs[self.rd_2_agent_id[key]] = sub_obs

        self.possible_agents = [f"agent_{i}" for i in range(self.num_routes * 2)]

        return all_obs, {}

    def get_updated_node_data(self):
        data = []

        for node in self.transit_system.topology.nodes:
            x = node.get_array()
            if node.associated_route != -1:
                buses_data = {}
                for route_id in self.nodes_in_routes.keys():
                    buses_data[route_id] = (
                        [
                            max(0, bus.capacity - len(bus.passengers))
                            for bus in self.transit_system.buses
                            if bus.service_route == route_id
                            and not bus.reversed
                            and node in bus.to_go
                        ],
                        [
                            max(0, bus.capacity - len(bus.passengers))
                            for bus in self.transit_system.buses
                            if bus.service_route == route_id
                            and bus.reversed
                            and node in bus.to_go
                        ],
                    )

                x = np.append(x, len(buses_data[node.associated_route][0]) / 10)
                x = np.append(x, len(buses_data[node.associated_route][1]) / 10)
                x = np.append(x, sum([0] + buses_data[node.associated_route][0]) / 10)
                x = np.append(x, sum([0] + buses_data[node.associated_route][1]) / 10)
                x = np.append(x, -1)
            else:
                x = np.append(x, 0)
                x = np.append(x, 0)
                x = np.append(x, 0)
                x = np.append(x, 0)
                x = np.append(x, -1)

            x = np.append(
                x,
                np.sin(
                    2
                    * np.pi
                    * self.current_time
                    / (self.hours_of_opperation_per_day * 3600)
                ),
            )
            data.append(x.astype(np.float32))

        return data

    def get_graph(self):
        graph = self.transit_system.topology.topology.copy()

        routes = self.transit_system.topology.routes
        if len(self.edge_data) == 0:
            for edge in graph.edges():
                for route in routes:
                    if route.node_pair_id == edge or route.node_pair_id[::-1] == edge:
                        self.edge_data[edge] = {"edge_attr": route.distance / 1000.0}
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
        if len(data) == 0:
            raise Exception("TransNET: Something went wrong in getting node data")
        obs.x = np.stack(data, axis=0)
        self.node_indices = {v: k for k, v in enumerate(self.graph.nodes)}
        return self.fix_obs_shape(obs)

    def del_data(self):
        self.edge_data = {}
        self.nodes_in_routes = {}
        self.current_day = 0

    def update_graph(self):
        data = self.get_updated_node_data()
        if len(data) == 0:
            raise Exception("TransNET: Something went wrong in getting node data")
        obs = from_networkx(self.graph)
        obs.x = np.stack(data, axis=0)
        return self.fix_obs_shape(obs)

    def fix_obs_shape(self, obs: Data, is_subgraph=False):
        if is_subgraph:
            max_edges = self.max_route_edges
            max_nodes = self.max_route_nodes
        else:
            max_edges = self.max_edges
            max_nodes = self.max_nodes

        if obs.edge_attr.ndim == 1:
            obs.edge_attr = np.expand_dims(obs.edge_attr, 1)

        # Pad node features (obs.x) along axis 0 to reach max_nodes
        pad_x = ((0, max_nodes - obs.x.shape[0]), (0, 0))  # pad rows
        obs.x = np.pad(obs.x, pad_x, mode="constant", constant_values=0)

        # Pad edge_index along axis 1 (columns) to reach max_edges
        pad_edge_index = (
            (0, 0),
            (0, max_edges - obs.edge_index.shape[1]),
        )  # pad columns

        obs.edge_index = np.pad(
            obs.edge_index, pad_edge_index, mode="constant", constant_values=0
        )

        # Pad edge_attr along axis 0 (rows) to reach max_edges
        pad_edge_attr = ((0, max_edges - obs.edge_attr.shape[0]), (0, 0))  # pad rows
        obs.edge_attr = np.pad(
            obs.edge_attr, pad_edge_attr, mode="constant", constant_values=0
        )

        return {"x": obs.x, "edge_index": obs.edge_index, "edge_attr": obs.edge_attr}

    def step(self, all_action: dict[str, int]):
        """
        Arguments:
        ---------
        `action`: contains the array of actions for each route id. Each array of the action consists of a list of two binary variables.
        these binary variables corresponds to each exit node of the route and the binary variable is to indicate whether to add a bus for that exit node or not.
        Since, a single model is used for all routes, The len of action can be changed from toplogy to toplogy but the mechanism will not fail.
        """

        for i, (agent_id, decision) in enumerate(all_action.items()):
            if decision == 1:
                self.transit_system.add_bus_on_route(
                    i // 2, reversed=False if i % 2 == 0 else True
                )

        reward, reward_info = self.reward(all_action)

        self.transit_system.step(self.current_time)

        if (
            self.current_time + self.analysis_period_sec
        ) >= self.hours_of_opperation_per_day * 3600:
            self.current_day += 1
            self.current_time = 0
            obs, _ = self.reset(hard_reset=False)
        else:
            self.current_time = self.current_time + self.analysis_period_sec

        truncated = {agent_id: False for agent_id in self.possible_agents}
        terminated = {agent_id: False for agent_id in self.possible_agents}
        
        if self.current_day >= self.analysis_period_days:
            for agent_id in self.possible_agents:
                truncated[agent_id] = True
                reward[agent_id] = 1000

        info = {**reward_info}

        for agent_id in self.possible_agents:
            if self.avg_waiting_time[agent_id] > 60:
                terminated[agent_id] = True
                reward[agent_id] = -1000

        obs: dict = self.update_graph()
        subgraphs = self.get_sub_graphs(obs)
        all_obs = {}

        for key, subgraph in subgraphs.items():
            sub_obs = {**obs, **{k + "_route": v for k, v in subgraph.items()}}
            all_obs[self.rd_2_agent_id[key]] = sub_obs

        reward = {k: reward[k] for k in all_obs}
        return all_obs, reward, terminated, truncated, info

    def get_sub_graphs(self, obs: dict) -> list[Data]:
        if obs["edge_index"].ndim == 2:
            subgraphs: dict = {}
            for (route_id, is_reversed), (
                indices,
                edge_index,
                edge_attr,
            ) in self.directed_sub_routes.items():
                sub_data = Data(
                    x=obs["x"][indices],
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=None if "y" not in obs else obs["y"][indices],
                )
                sub_data["x"][:, -1] = float(is_reversed)
                subgraphs[(route_id, is_reversed)] = self.fix_obs_shape(
                    sub_data, is_subgraph=True
                )
        else:
            raise Exception("Higher batch is processed to create subgraphs")

        return subgraphs

    def reward(self, actions) -> float:
        rewards = {}
        rewards_info = {}

        for i, (agent_id, action) in enumerate(actions.items()):
            route_id = int(i // 2)
            is_reversed = not (i % 2 == 0)

            num_passengers = []

            for node in self.transit_system.topology.nodes:
                if route_id in node.affliated_route_ids:
                    passengers = [p for p in node.passengers if p.is_reversed == is_reversed]
                    num_passengers.append(len(passengers))
                    
            avg_waiting_time = [
                np.max([0] + [passenger.waiting_time for passenger in node.passengers if passenger.is_reversed == is_reversed])
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
                np.max([0] + [passenger.stranding_counts for passenger in node.passengers if passenger.is_reversed == is_reversed])
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
                avg_waiting_time = np.max(avg_waiting_time) / 60.0  # minutes
            else:
                avg_waiting_time = 0.0

            if len(avg_stranding_count) > 0:
                avg_stranding_count = np.max(avg_stranding_count)
            else:
                avg_stranding_count = 0  # counts

            reward_2 = 0
            reward_2 += -(avg_waiting_time // 5)

            if avg_stranding_count > 0 and action == 0:
                reward_2 += -2

            self.avg_waiting_time[self.possible_agents[i]] = avg_waiting_time

            if action == 1:
                expence_of_bus_journey = 1
            else:
                expence_of_bus_journey = 0

            reward_3 = -expence_of_bus_journey


            reward = reward_3

            buses = [bus for bus in self.transit_system.retired_buses if bus.service_route == route_id and bus.reversed == is_reversed]
            reward += sum([bus.num_passengers_served > 0 for bus in buses])
            for bus in self.transit_system.retired_buses:
                self.transit_system.retired_buses.remove(bus)

            reward_info = {
                "reward_type_2": reward_2,
                "reward_type_3": reward_3,
                "reward": reward,
            }

            rewards[agent_id] = reward
            rewards_info[agent_id] = reward_info

        pd.DataFrame(rewards_info.values()).mean().to_dict()
        return rewards, rewards_info

    def render(self):
        pass
