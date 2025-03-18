import numpy as np
from topology import Topology
from bus import Bus
from logger import PassengerLogger

class TransitSystem:
    "Representation of a public transit system"
    def __init__(self,
                 num_busses_per_route: int = 1,
                 min_bus_capacity: int = 100, 
                 max_bus_capacity: int = 200,
                 avg_bus_speed: float = 16.67,
                 seed: int = 0
                 ) -> None:
        """
        Argument:
        --------
        `num_busses_per_route` is a fixed max number busses per route
        `[min,max]_bus_capacity` is the [minimum, maximum] number of people a single bus can hold
        `avg_bus_speed`is the speed in m/s
        `seed`: is for generating a random scenerio
        """
        np.random.seed(seed=seed)
        self.topology = Topology()

        self.num_busses_per_route = num_busses_per_route
        self.capacity = np.random.randint(min_bus_capacity, max_bus_capacity)
        self.buses = [([Bus(self.capacity, 
                          avg_bus_speed, 
                          i, 
                          avg_bus_speed, 
                          self.topology,
                          reversed=False,
                          ) for _ in range(num_busses_per_route)] +

                      [Bus(self.capacity, 
                          avg_bus_speed, 
                          i, 
                          avg_bus_speed, 
                          self.topology,
                          reversed=True,
                          ) for _ in range(num_busses_per_route)]) for i in range(len(set([r.route_id for r in self.topology.routes])))]
    
        self.buses = sum(self.buses, [])
        self.passenger_logger = PassengerLogger("passenger_logs.csv")
        
    def step(self, time) ->None:
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
                self.passenger_logger.add_to_pool(**passenger.to_dct())
                self.passenger_logger.commit()
