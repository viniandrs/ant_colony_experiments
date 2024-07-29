import numpy as np

class TSP:
    def __init__(self, cities):
        self.cities = cities
        self.cost_matrix = self._calculate_cost_matrix()

    @classmethod
    def from_instance(cls, cities):
        return cls(cities)
    
    @classmethod
    def random_instance(cls, n_cities):
        return cls(np.random.rand(n_cities, 2))
    
    def _calculate_cost_matrix(self):
       # calculate the cost matrix with the distances between cities i and j
       z = np.array([[complex(*c) for c in self.cities]])
       return abs(z.T - z) + 1e-8
    
    def evaluate(self, tours):
        """
        Compute the total cost of a batch of solutions

        Parameters
        ----------
        tours : np.array
            An array of shape (n_tours, n_cities) containing a set of tours.
            A tour is a list of cities visited by the agent.
        """
        tour_costs = np.zeros(tours.shape[0])
        for n in range(tours.shape[-1] - 1): # iterating on each edge walked bythe ants on their tour (except the last one)
            tour_costs += self.cost_matrix[tours[:, n], tours[:, n + 1]]
        
        return tour_costs