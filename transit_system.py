import numpy as np
from topology import Topology
from bus import Bus
from passenger import Passenger
from logger import PassengerLogger


class TransitSystem:
    "Representation of a public transit system"

    def __init__(
        self,
        num_busses_per_route: int = 1,
        min_bus_capacity: int = 100,
        max_bus_capacity: int = 200,
        avg_bus_speed: float = 16.67,
        analysis_period_sec: float = 60,
        min_num_stops_per_route: int = 8,
        max_num_stops_per_route: int = 32,
        min_num_route_per_toplogy: int = 4,
        max_num_route_per_toplogy: int = 12,
        hours_of_opperation_per_day: int = 18,
        mean_population_density: float = 2000.0,
        std_population_density: float = 500.0,
        min_population_density: float = 350.0,
        mean_catchment_radius: float = 2.0,
        std_catchment_radius: float = 1.0,
        min_catchment_radius: float = 1.0,
        min_transit_users_proportion: float = 0.05,
        max_transit_users_proportion: float = 0.30,
        min_distance: float = 500.0,
        max_distance: float = 2000.0,
        log_passengers: bool = True,
        seed: int = 0,
        *args,
        **kwargs,
    ) -> None:
        """
        Argument:
        --------
        `num_busses_per_route` is a fixed max number busses per route per direction
        `[min,max]_bus_capacity` is the [minimum, maximum] number of people a single bus can hold
        `avg_bus_speed`is the speed in m/s
        `analysis_period_sec` : is the least count of time
        `[min,max]_num_stops_per_route` : is the upper and lower bound of nodes for each route
        `[min,max]_num_route_per_toplogy` : is the upper and lower bound of routes for each topology
        `hours_of_opperation_per_day` : is the maximum hours the buses will keep running
        `analysis_period_sec` : is the least count of time
        `mean_population_density`: is the mean population density of catchment area of each station
        `std_population_density`: is the standrd deviation of population density of catchment area of each station
        `min_population_density`: is the lowerbound to clip smaller values for the population density of catchment area of each station
        `mean_catchment_radius`: is the mean area of the catchment area of each station
        `std_catchment_radius`: is the standard deviation area of the catchment area of each station
        `min_catchment_radius`: is the lowerbound to clip smaller values for the area of catchment area of each station
        `min_transit_users_proportion`: is the minimum ratio of transit users given the population for a station
        `max_transit_users_proportion`: is the maximum ration of transit users given the population for a station
        `min_distance`: is the minumum distance between neighbouring nodes
        `max_distance`: is the maximum distance between neighbouring nodes
        `seed`: is for generating a random scenerio
        """
        self.seed = seed
        self.analysis_period_sec = analysis_period_sec
        np.random.seed(seed=seed)
        self.topology = Topology(
            analysis_period_sec=analysis_period_sec,
            min_num_stops_per_route=min_num_stops_per_route,
            max_num_stops_per_route=max_num_stops_per_route,
            min_num_route_per_toplogy=min_num_route_per_toplogy,
            max_num_route_per_toplogy=max_num_route_per_toplogy,
            hours_of_opperation_per_day=hours_of_opperation_per_day,
            mean_population_density=mean_population_density,
            std_population_density=std_population_density,
            min_population_density=min_population_density,
            mean_catchment_radius=mean_catchment_radius,
            std_catchment_radius=std_catchment_radius,
            min_catchment_radius=min_catchment_radius,
            min_transit_users_proportion=min_transit_users_proportion,
            max_transit_users_proportion=max_transit_users_proportion,
            min_distance=min_distance,
            max_distance=max_distance,
            seed=seed,
        )

        self.num_busses_per_route = num_busses_per_route
        self.capacity = np.random.randint(min_bus_capacity, max_bus_capacity)
        self.avg_bus_speed = avg_bus_speed
        self.analysis_period_sec = analysis_period_sec
        self.buses: list[Bus] = []

        self.log_passengers = log_passengers
        self.passenger_logger = PassengerLogger("logs")

        self.num_busses_added = 0
        self.num_buses_done = 0
        self.num_passengers_done = 0

        self.route_ids = self.topology.route_ids
        for route_id in self.route_ids:
            for _ in range(self.num_busses_per_route):
                self.add_bus_on_route(route_id, reversed=False)
                self.add_bus_on_route(route_id, reversed=True)
        
        self.retired_buses = set()
        self.step_retired_buses = set()

        self.report = {
            "total_deployed_buses": [],
            "total_done_buses": [],
            "total_done_passengers": [],
            "total_avg_waiting_time": [],
            "total_avg_average_stranded_count": [],
            **{
                f"route_{route_id}_{is_reversed}": {
                    "total_deployed_buses": [],
                    "total_done_buses": [],
                    "total_done_passengers": [],
                    "total_avg_waiting_time": [],
                    "total_avg_average_stranded_count": [],
                }
                for route_id in self.route_ids
                for is_reversed in [True, False]
            }
        }

    def add_bus_on_route(self, route_id: int, reversed: bool, time: int):
        """
        Appends a bus to a route id.

        Argument:
        --------
        `route_id`: the route id on which the bus will opperate
        `reversed`: the state of the bus i.e, goung from A-->B or B-->A
        """
        self.buses.append(
            Bus(
                self.capacity,
                self.avg_bus_speed,
                route_id,
                self.analysis_period_sec,
                self.topology,
                reversed=reversed,
                analysis_period_sec=self.analysis_period_sec,
                created_at=time,
            )
        )
        self.report["total_deployed_buses"].append(1)
        self.report[f"route_{route_id}_{reversed}"]["total_deployed_buses"].append(1)
        self.num_busses_added += 1

    def claculate_passenger_parametres(self, time: int, passenger: Passenger):
        """
        Argument:
        --------
        `time`: is the time is seconds starting from the first hour of the opperation to the last hour of opperation
        `passenger`: instance of `Passenger` class that has finished journey
        """
        if passenger.travel_time != 0:
            path = passenger.path
            distances = []
            for u, v in zip(path[:-1], path[1:]):
                for route in self.topology.routes:
                    if u.node_id in route.node_pair_id and v.node_id in route.node_pair_id:
                        distances.append(route.distance)

            passenger.distance_traversed = np.sum(distances)
            passenger.total_time_taken = time - passenger.started_at
            passenger.average_travel_speed = (
                passenger.distance_traversed / passenger.travel_time
            )

    def step(self, time) -> None:
        """
        It calls the step of nodes and busses

        Argument:
        --------
        `time`: is the time is seconds starting from the first hour of the opperation to the last hour of opperation

        """
        od_mat = self.topology.get_od_mat_for_time(time)
        for i, node in enumerate(self.topology.nodes):
            node.step(time, to_depart=od_mat[i, :], all_nodes=self.topology.nodes)

        for i, bus in enumerate(self.buses):
            passengers = bus.step(time)
            for passenger in passengers:
                self.num_passengers_done += 1
                self.claculate_passenger_parametres(time, passenger)
                self.report["total_done_passengers"].append(1)
                self.report[f"route_{bus.service_route}_{bus.reversed}"]["total_done_passengers"].append(1)
                self.report["total_avg_waiting_time"].append(passenger.tagged_waiting_time)
                self.report[f"route_{bus.service_route}_{bus.reversed}"]["total_avg_waiting_time"].append(passenger.tagged_waiting_time)
                
                if self.log_passengers:
                    self.passenger_logger.add_to_pool(
                        seed=self.seed, time=time, **passenger.to_dct()
                    )
                    self.passenger_logger.commit()
        
        to_drop = []
        for bus in self.buses:
            if bus.done:
                to_drop.append(bus)
                self.step_retired_buses.add(bus)
                self.retired_buses.add(bus)
                self.report["total_done_buses"].append(1)
                self.report[f"route_{bus.service_route}_{bus.reversed}"]["total_done_buses"].append(1)
                

        for bus in to_drop:
            if bus in self.buses:
                self.buses.remove(bus)

        self.num_buses_done += len(to_drop)
        