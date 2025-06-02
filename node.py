from __future__ import annotations
import numpy as np
from typing import Self, TYPE_CHECKING
from passenger import Passenger
from functions import softmax

if TYPE_CHECKING:
    from bus import Bus


class Node:
    """
    This class represents the transit station. It has following features
    1- Population density (chosen randomly using normal distribution with mu=300 persons/km and std=200 persons/km and the lower bound is clipped with min=100)
    2- Area (chosen randomly using normal distribution with mean = 2 km radius and std = 1 square km and the lower bound is clipped with min=0.5 km radius)
    3- Population (calculated using population density)
    4- The proportion of Transit users (randomly assigned using a modified probability distribution)

    Other parametres:
    ----------------
    `affiliated routes` is the set of routes that contain this `Node`
    `zone type` is used by the `Topology` class to produce the OD matrix for departures and arrivals
    """

    def __init__(
        self,
        node_id: int | None = None,
        mean_population_density: float = 300.0,
        std_population_density: float = 200.0,
        min_population_density: float = 100.0,
        mean_catchment_radius: float = 2.0,
        std_catchment_radius: float = 1.0,
        min_catchment_radius: float = 0.5,
        min_transit_users_proportion: float = 0.05,
        max_transit_users_proportion: float = 0.3,
        analysis_period_sec: int = 60,
    ) -> None:
        """
        Argument:
        --------

        `node_id` is the integer value and the value of the station as represented in the `Topology.topology : nx.Graph`
        `mean_population_density` is the mean population density of catchment area of each station
        `std_population_density` is the standrd deviation of population density of catchment area of each station
        `min_population_density` is the lowerbound to clip smaller values for the population density of catchment area of each station
        `mean_catchment_radius` is the mean area of the catchment area of each station
        `std_catchment_radius` is the standard deviation area of the catchment area of each station
        `min_catchment_radius` is the lowerbound to clip smaller values for the area of catchment area of each station
        `min_transit_users_proportion` is the minimum ratio of transit users given the population for a station
        `max_transit_users_proportion` is the maximum ration of transit users given the population for a station
        `analysis_period_sec` is the time period in seconds for which the simulation is run
        """
        self.node_id = int(node_id)
        self.mean_population_density = mean_population_density
        self.mean_catchment_radius = mean_catchment_radius
        self.analysis_period_sec = analysis_period_sec

        self.population_density_ppkm2 = max(
            np.random.normal(loc=mean_population_density, scale=std_population_density),
            min_population_density,
        )
        self.catchment_area_km2 = max(
            np.random.normal(
                loc=mean_catchment_radius**2 * np.pi,
                scale=std_catchment_radius**2 * np.pi,
            ),
            min_catchment_radius**2 * np.pi,
        )
        self.population = self.population_density_ppkm2 * self.catchment_area_km2
        self.population_proportion_using_transit = np.random.choice(
            np.linspace(
                min_transit_users_proportion, max_transit_users_proportion, 1000
            ),
            p=softmax(np.linspace(4, -4, 1000), axis=0),
        )

        self.transit_users: float = self.population * self.population_proportion_using_transit

        self.affiliated_routes = set() #all the routes the node is attached to (can provide info on the associated route id and if it's transfer)
        self.affliated_route_ids = set()
        self.exit_nodes : list[int] = []
        self.is_exit = False
        self.is_transfer = False
        self.zone_type = "residentials"
        self.od_route = {} #shortest path to all the other nodes in the topology
        self.od_distance: dict[int, float] = {}
        self.temp_waiting_passengers = {}

        self.passengers: list[Passenger] = []

        self.arrivals: list[int] = [0]
        self.departures: list[int] = [0]
        self.ins_arrivals: int = 0
        self.ins_departures: int = 0
        self.time_of_last_bus_0: int = 0
        self.time_of_last_bus_1: int = 0

        self.step_counter: int = 1
        self.associated_route = -1

        self.avg_waiting_time_0 = 0
        self.avg_waiting_time_1 = 0
        self.avg_stranding_counts_0 = 0
        self.avg_stranding_counts_1 = 0
        

        self.zone_type_id = {k:e for e, k in enumerate(["school", "office", "shopping", "residentials"])}[self.zone_type]

    def check_transfers(self, destination: Self) -> list[Self]:
        """
        It takes in the destination node and calculates the trajectory to be followed.
        It finally extract the transfer nodes and return a list of transfers.

        Argument:
        --------
        `destination` is a `Node` for where the used wants to go

        Return:
        ------
        A List of `Nodes` that will work as transfer nodes in this travel
        """
        path = self.od_route[destination.node_id]

        transfers = []
        for node in path:
            if node.is_transfer and node != self and node != destination:
                transfers.append(node)

        return transfers

    def step(
        self, time: int, to_depart: np.ndarray, all_nodes: dict[int, Self]
    ) -> None:
        """
        A step function that will be called repeatedly as the time progresses.
        This method performs following functionalities

        1- Generate passengers based on th `to_depart` which is a function of time
        2- Create a `Passenger` instance and initilize precalculate it trajectory/journey

        Argument:
        --------
        `time`: is the time is seconds starting from the first hour of the opperation to the last hour of opperation
        `to_depart`: is calculted by the `Topology` object and is a od-matrix with origin being this `Node` and destination being all the other `Node`s is `Topology`
        `all_nodes`: is a dict of all the `Node`s in the repo with `node_id` as key and `Node` itself as value

        """
        self.ins_arrivals = 0
        self.ins_departures = 0

        for i in range(len(to_depart)):
            if i not in self.temp_waiting_passengers:
                self.temp_waiting_passengers[i] = 0
            self.temp_waiting_passengers[i] += to_depart[i]

        num_passengers = {
            k: int(v) for k, v in self.temp_waiting_passengers.items() if int(v) > 0
        }
        for k, v in num_passengers.items():
            if v > 0:
                self.temp_waiting_passengers[k] -= v
                for _ in range(v):
                    path = self.od_route[all_nodes[k].node_id]
                    if self != all_nodes[k]:
                        self.ins_arrivals += 1
                        self.passengers.append(
                            Passenger(
                                origin=self,
                                destination=all_nodes[k],
                                queued_since=time,
                                transfers=self.check_transfers(all_nodes[k]),
                                path=path,
                            )
                        )

        for passenger in self.passengers:
            passenger.step(self)
            
        for passenger in self.passengers:
            passenger.waiting_time += self.analysis_period_sec
            
        self.step_counter += self.analysis_period_sec
        self.avg_waiting_time_0 = np.max([0]+[passenger.waiting_time for passenger in self.passengers if not passenger.is_reversed])
        self.avg_waiting_time_1 = np.max([0]+[passenger.waiting_time for passenger in self.passengers if passenger.is_reversed])
        self.avg_stranding_counts_0 = np.max([0]+[passenger.stranding_counts for passenger in self.passengers if not passenger.is_reversed])
        self.avg_stranding_counts_1 = np.max([0]+[passenger.stranding_counts for passenger in self.passengers if passenger.is_reversed])
        

    def bus_arrived(self, time: int, bus: Bus) -> list[Passenger]:
        """
        This function does the following:
        1- transfer passengers having destination or tranfer in the `bus.to_go` to the `bus`
        2- change the status i.e, `waiting time`, `is aboard` etc., of the passengers accordingly
        3- transfer passengers from the `bus` to the `Node` if the `Node` is destination of transfer

        Argument:
        --------
        `time`: is the time is seconds starting from the first hour of the opperation to the last hour of opperation
        `bus`: is the instance of the bus thatust crosses this `Node`

        Returns:
        list of passengers that have reached destination
        """
        
        to_drop : list[Passenger] = []
        to_drop_from_bus: list[Passenger] = []

        for passenger in bus.passengers:
            if passenger.destination == self:
                passenger.queued_since = time
                to_drop.append(passenger)
                
            else:
                to_drop_first_transfer: list[Passenger] = []
                for transfer in passenger.transfers:
                    if transfer == self:
                        passenger.queued_since = time
                        self.passengers.append(passenger)
                        to_drop_from_bus.append(passenger)
                        to_drop_first_transfer.append(passenger)

                for passenger in to_drop_first_transfer:
                    passenger.transfers.remove(self)

        for passenger in to_drop + to_drop_from_bus:
            if passenger in bus.passengers:
                bus.passengers.remove(passenger)
                self.ins_arrivals += 1

        aboard: list[Passenger] = []
        for passenger in self.passengers:
            if len(bus.passengers) < bus.capacity:
                if passenger.destination in bus.to_go:
                    passenger.queued_since = time
                    bus.passengers.append(passenger)
                    aboard.append(passenger)
                else:
                    for transfer in passenger.transfers:
                        if transfer in bus.to_go:
                            passenger.queued_since = time
                            bus.passengers.append(passenger)
                            aboard.append(passenger)
            else:
                if passenger.destination in bus.to_go:
                    passenger.stranding_counts += 1
                else:
                    for transfer in passenger.transfers:
                        if transfer in bus.to_go:
                            passenger.stranding_counts += 1

        for passenger in aboard:
            if passenger in self.passengers:
                passenger.waiting_time = 0 # have corrupted the log
                passenger.stranding_counts = 0 # have corrupted the log
                self.passengers.remove(passenger)
                self.ins_departures += 1

        if not bus.reversed:
            self.time_of_last_bus_0 = time
        else:
            self.time_of_last_bus_1 = time
        return to_drop

    def distance_to_exit_nodes(self):
        distances = []
        for node_id in self.exit_nodes:
            if node_id == self.node_id:
                distances.append(0)
            else:
                distances.append(self.od_distance[node_id])
                
        return distances
    

    def get_dct(self) -> dict:
        """
        This function returns the dictionary of the `Node` object
        containing the following information:
        1- population density
        2- catchment area
        3- population
        4- proportion of transit users
        5- is transfer
        6- min distance from the exit node of the route
        7- max distance from the exit node of the route (in case of transfer node all the connecting routes will be considered)
        8- average arrivals
        9- average departures
        10- average waiting time
        11- average stranding counts
        12- time elapsed since last bus
        13- number of waiting passengers
        14- number of stranding passengers
        """
        self.arrivals.append(self.ins_arrivals)
        self.departures.append(self.ins_departures)

        return {
            "population_density": self.population_density_ppkm2 / 1000,
            "catchment_area": self.catchment_area_km2 / (np.pi * 4) ,
            "population": self.population / (np.pi * 4 * 1000),
            "transit_users": self.transit_users / (np.pi * 4 * 1000),
            "is_transfer": float(self.is_transfer),
            "min_distance_from_exit_node": min(self.distance_to_exit_nodes()) / 3000.0,
            "max_distance_from_exit_node": max(self.distance_to_exit_nodes()) / 3000.0,
            "average_arrivals": np.mean(self.arrivals[-10:])/100.,
            "average_departures": np.mean(self.departures[-10:])/100.,
            "average_waiting_time_0": self.avg_waiting_time_0/60,
            "average_stranding_counts_0": self.avg_stranding_counts_0,
            "average_waiting_time_1": self.avg_waiting_time_1/60,
            "average_stranding_counts_1": self.avg_stranding_counts_1,
            "time_elapsed_since_last_bus_0": (self.step_counter - self.time_of_last_bus_0) / 3600.,
            "time_elapsed_since_last_bus_1": (self.step_counter - self.time_of_last_bus_1) / 3600.,
            "number_of_waiting_passengers_0": len([p for p in self.passengers if not p.is_reversed]) / 10,
            "number_of_stranding_passengers_1": len([passenger for passenger in self.passengers if passenger.stranding_counts>0 and not passenger.is_reversed]) / 10,
            "number_of_waiting_passengers_1": len([p for p in self.passengers if p.is_reversed]) / 10,
            "number_of_stranding_passengers_0": len([passenger for passenger in self.passengers if passenger.stranding_counts>0 and passenger.is_reversed]) / 10,
            "zone_type": self.zone_type_id
        }
    
    def get_array(self) -> list:
        """
        This function returns the array of the `Node` object
        containing the following information:
        1- population density
        2- catchment area
        3- population
        4- proportion of transit users
        5- is transfer
        6- min distance from the exit node of the route
        7- max distance from the exit node of the route (in case of transfer node all the connecting routes will be considered)
        8- average arrivals
        9- average departures
        10- average waiting time
        11- average stranding counts
        12- time elapsed since last bus
        13- number of waiting passengers
        14- number of stranding passengers
        15- zone type
        16- time
        """

        return np.array(list(self.get_dct().values()))


    def __repr__(self):
        """
        Override the to_string functionality
        """
        return f"Node {self.node_id} population: {int(self.population)} transit users: {int(self.transit_users)}"
