from __init__ import *
from passenger import Passenger

class Node:
    def __init__(self,
                 node_id=None) -> None:
        
        self.node_id = int(node_id)
        self.population_density_ppkm2 = max(np.random.normal(loc=300, scale=200), 100)
        self.catchment_area_km2 = np.random.normal(loc = 4*np.pi, scale = np.pi)
        self.population = self.population_density_ppkm2 * self.catchment_area_km2
        self.population_proportion_using_transit = np.random.choice(np.linspace(0.05, 0.3, 1000), 
                                                                    p=softmax(np.linspace(4, -4, 1000), axis=0))
        
        self.transit_users = self.population * self.population_proportion_using_transit
        
        self.affiliated_routes = set()
        self.is_exit = False
        self.is_transfer= False
        self.zone_type = "residential"
        self.od_route = {}
        self.temp_waiting_passengers = {}

        self.stranding_passengers = 0
        self.average_waiting_time = 0
        self.passengers = []
        
    # def check_transfers(self, destination):
    #     path = self.od_route[destination.node_id]        
    #     affiliated_routes_in_path = [node.affiliated_routes for node in path]

    #     affliated_route_ids = []
    #     for set_of_routes in affiliated_routes_in_path:
    #         if len(set_of_routes) > 2:
    #             affliated_route_ids.append(None)
    #         else:
    #             route = list(set_of_routes)[0]
    #             affliated_route_ids.append(route.route_id)
            
    #     num_continuous_nones = []
    #     for iid in affliated_route_ids[::-1]:
    #         if iid is None:
    #             num_continuous_nones.append(iid)
    #         else:
    #             break

    #     num_continuous_nones = len(num_continuous_nones)

    #     while None in affliated_route_ids[:-num_continuous_nones]:
    #         for i in range(len(affliated_route_ids)):
    #             if affliated_route_ids[i] is None:
    #                 if i+1 < len(affliated_route_ids):
    #                     affliated_route_ids[i] = affliated_route_ids[i+1]
        
    #     if num_continuous_nones > 0:
    #         iid = affliated_route_ids[:-num_continuous_nones][-1]
    #         for i in range(1,num_continuous_nones+1):
    #             affliated_route_ids[-i] = iid
        
    #     is_transfer = np.zeros(len(path))
    #     for i in range(len(affliated_route_ids)-1):
    #         if affliated_route_ids[i] != affliated_route_ids[i+1]:
    #             is_transfer[i+1] = 1

    #     transfers = []
    #     for i, it in enumerate(is_transfer):
    #         if it:
    #             transfers.append(path[i])
                
    #     return transfers

    def check_transfers(self, destination):
        path = self.od_route[destination.node_id]        
        
        # Extract affiliated route IDs or None if more than 2 routes exist
        affliated_route_ids = [
            list(routes)[0].route_id if len(routes) <= 2 else None 
            for routes in (node.affiliated_routes for node in path)
        ]

        # Forward fill `None` values
        last_valid = None
        for i in range(len(affliated_route_ids)):
            if affliated_route_ids[i] is None:
                affliated_route_ids[i] = last_valid
            else:
                last_valid = affliated_route_ids[i]

        # Backward fill if the tail had `None`s
        last_valid = None
        for i in range(len(affliated_route_ids) - 1, -1, -1):
            if affliated_route_ids[i] is None:
                affliated_route_ids[i] = last_valid
            else:
                last_valid = affliated_route_ids[i]

        # Detect transfers (change in route IDs)
        transfers = [path[i] for i in range(1, len(affliated_route_ids)) 
                    if affliated_route_ids[i] != affliated_route_ids[i-1]]

        return transfers

    def step(self, time, to_depart, all_nodes):
        for i in range(len(to_depart)):
            if i not in self.temp_waiting_passengers:
                self.temp_waiting_passengers[i] = 0
            self.temp_waiting_passengers[i] += to_depart[i]

        num_passengers = {k:int(v) for k,v in self.temp_waiting_passengers.items() if int(v) > 0}
        for k, v in num_passengers.items():
            if v > 0:
                self.temp_waiting_passengers[k] -= v
                for _ in range(v):
                    self.passengers.append(Passenger(origin=self, 
                                                     destination=all_nodes[k], 
                                                     queued_since=time,
                                                     transfers=self.check_transfers(all_nodes[k])))
                    
    def bus_arrived(self, time, bus):
        to_drop = []
        for passenger in bus.passengers:
            if passenger.destination == self:
                passenger.is_dropped = True
                passenger.travel_time += time - passenger.queued_since
                passenger.queued_since = time
                passenger.is_aboard = False
                to_drop.append(passenger)
            else:
                for transfer in passenger.transfers:
                    if transfer == self:
                        passenger.is_dropped = True
                        passenger.travel_time += time - passenger.queued_since
                        passenger.queued_since = time
                        passenger.is_aboard = False
                        to_drop.append(passenger)
                        self.passengers.append(passenger)
                        passenger.transfers.remove(transfer)

        for passenger in to_drop:
            bus.passengers.remove(passenger)
        
        for passenger in self.passengers:
            if len(bus.passengers) < bus.capacity:
                if passenger.destination in bus.to_go:
                    passenger.is_aboard = True
                    passenger.waiting_time += time - passenger.queued_since
                    passenger.queued_since = time
                    bus.passengers.append(passenger)
                    passenger.is_dropped = False
                else:
                    for transfer in passenger.transfers:
                        if transfer in bus.to_go:
                            passenger.is_aboard = True
                            passenger.waiting_time += time - passenger.queued_since
                            passenger.queued_since = time
                            bus.passengers.append(passenger)
                            passenger.is_dropped = False
                
    def __repr__(self):
        return f"Node {self.node_id} population: {int(self.population)} transit users: {int(self.transit_users)}"
