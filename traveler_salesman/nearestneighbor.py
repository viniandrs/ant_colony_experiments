import numpy as np
from tsp import TSP

class NearestNeighborSolver:
    @classmethod
    def solve(cls, tsp_instance: TSP):
        """
        Solve the TSP instance using the nearest neighbor algorithm.
        """
        n_cities = tsp_instance.cities.shape[0]
        distance_matrix = tsp_instance.cost_matrix
        visited = np.zeros(n_cities, dtype=bool)
        tour = []

        current_city = np.random.randint(n_cities)
        visited[current_city] = True
        tour.append(current_city)
        for _ in range(n_cities - 1):
            next_city = np.argmin(distance_matrix[current_city] + 1e6 * visited)
            visited[next_city] = True
            tour.append(next_city)
            current_city = next_city
            
        return np.array([tour])
        