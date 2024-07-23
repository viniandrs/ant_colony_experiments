import numpy as np
import matplotlib.pyplot as plt

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

    def get_cost(self, tours):
        """
        Compute the total cost of a batch of solutions

        Parameters
        ----------
        tours : np.array
            An array of shape (n_tours, n_cities) containing a set of tours.
            A tour is a list of cities visited by the agent.
        """
        n_tours = tours.shape[0]
        n_cities = tours.shape[1]
        tour_costs = np.zeros(n_tours)
        for n in range(n_cities): # iterating on each edge walked bythe ants on their tour
            tour_costs += self.cost_matrix[tours[:, n % n_cities], tours[:, (n + 1) % n_cities]]
        
        return tour_costs

    def render(self, tour=None):
        plt.figure()
        plt.scatter(self.cities[:,0],self.cities[:,1])
        
        if tour is None:
            plt.gca().set_aspect('equal', adjustable='box')
            return

        for a, b in zip(tour[:-1], tour[1:]):
            x = self.cities[[a, b]].T[0]
            y = self.cities[[a, b]].T[1]
            plt.plot(x, y, c='r', zorder=-1)

        plt.gca().set_aspect('equal')
        print('Tour length: ', self.get_cost(np.expand_dims(tour, 0)))

        