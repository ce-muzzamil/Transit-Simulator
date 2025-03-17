from __init__ import *
from topology import Topology
from bus import Bus

class TransitSystem:
    def __init__(self,
                 num_busses_per_route=1,
                 min_bus_capacity=100, 
                 max_bus_capacity=200,
                 avg_bus_speed=16.67,
                 seed=0
                 ) -> None:
        
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
        
    def step(self, time):
        od_mat = self.topology.get_od_mat_for_time(time)
        for i, node in enumerate(self.topology.nodes):
            node.step(time, to_depart=od_mat[i, :], all_nodes=self.topology.nodes)
        
        for i, bus in enumerate(self.buses):
            bus.step(time)