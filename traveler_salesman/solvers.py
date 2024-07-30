import numpy as np
from tsp import TSP
from nearestneighbor import NearestNeighborSolver

from typing import Tuple

class AntSolver:
    def __init__(
        self, 
        tsp_instance: TSP, 
        alpha=0.1, # pheromone decay
        beta=2, # importance of the pheromone over the quality of the edge
        n_ants=10, # number of ants
        ):
        self.instance = tsp_instance
        self.distance_matrix = tsp_instance.cost_matrix
        self.n_cities = tsp_instance.cities.shape[0]
        self.alpha = alpha 
        self.beta = beta
        self.n_ants = min(n_ants, self.n_cities)
        
        Lnn = tsp_instance.evaluate(NearestNeighborSolver.solve(tsp_instance)) # length of the nearest neighbor solution
        self.initial_phero_level = 1 / (self.n_cities * Lnn)
        self.pheromones = np.full((self.n_cities, self.n_cities), self.initial_phero_level) # initialize pheromone matrix

    def sample(self, n_tours):
        """
        Sample n_tours from the TSP instance.
        """
        pass

    def _initialize_ants(self, n_ants=None):
        """
        Initialize the ants at random cities.
        """
        if n_ants is None:
            n_ants = self.n_ants
        elif n_ants > self.n_cities:
            raise ValueError("The number of ants should be less than the number of cities.")
        
        return np.random.choice(self.n_cities, n_ants)

    def _transition_probability(self, current_city, visited_cities):
        """
        calculate the vector of transition probabilities from the current city to the unvisited cities.
        """
        mask = ~np.isin(np.arange(self.n_cities), visited_cities)
        
        weights = np.zeros(self.n_cities)
        weights[mask] = self.pheromones[current_city, mask] / (self.distance_matrix[current_city, mask] ** self.beta)

        # normalizing the weights to get the probabilities
        probs = weights/np.sum(weights, axis=-1, keepdims=True) 

        return probs

    def update_pheromone(self, tours):
        """
        Update the pheromone matrix based on a set of tours.
        """
        pass

class AntSystemSolver(AntSolver):
    """
    Ant System is the progenitor of all ACO algorithms. It is a simple algorithm that uses a 
    not-optimized general-purpose ant colony optimization strategy.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def sample(
        self,
        n_tours=None,
        strategy='stochastic'
        ):
        assert n_tours is None or n_tours <= self.n_ants, 'Sampling for n_tours > n_ants is not supported'
        if n_tours is None:
            n_tours = self.n_ants

        starting_positions = self._initialize_ants(n_tours)
        tours = []
        for ant in range(n_tours):
            starting_position = starting_positions[ant]
            probs = self._transition_probability(starting_position, visited_cities=[starting_position])
            
            if strategy == 'stochastic':
                tour = np.concatenate(([starting_position], np.random.choice(np.arange(self.n_cities), 
                                                                replace=False, size=self.n_cities-1, p=probs)))
            elif strategy == 'greedy':
                zeros = np.where(probs == 0)[0]
                tour = np.concatenate(([starting_position], np.argsort(-probs)[:-len(zeros)]))

            tours.append(tour)

        return np.array(tours)
    
    def update_pheromone(self, tours):
        tour_lengths = self.instance.evaluate(tours)

        """
        The for loop below calculates the delta_phero matrix for each of the k ants with the best tours.
        The delta_phero matrix has 1/tour_length[ant] in the position (i, j) if the ant visited the city j after the city i
        and 0 otherwise.
        """
        delta_pheros = np.zeros((self.n_ants, self.n_cities, self.n_cities))
        for ant in range(self.n_ants):
            tour = tours[ant]
            tour = np.concatenate((tour, [tour[0]])) 
            edges = np.array([[city, next_city] for city, next_city in zip(tour[:-1], tour[1:])])

            # the i-th element of the array below is the city visited after city i
            next_cities = edges[edges[:, 0].argsort()][:, 1]
            
            # scattering ones in the positions that correspond to edges walked by the ant
            values = np.ones(delta_pheros.shape[1])
            values[np.where(next_cities == tour[0])[0]] = 0 # discarding the edge which closes the loop
            np.put_along_axis(delta_pheros[ant], np.expand_dims(next_cities, 1), np.expand_dims(values, 1), axis=-1)
            delta_pheros[ant] = (delta_pheros[ant] + delta_pheros[ant].T) / tour_lengths[ant] # making the matrix symmetric and dividing by the tour length

        # applying the pheromone update rule 
        self.pheromones = (1 - self.alpha) * self.pheromones + np.sum(delta_pheros, axis=0)

class AntColonySystemSolver(AntSolver):
    """
    Ant Colony System (ACS) is an extension of the Ant System algorithm speciallized in
    solving TSPs with improved efficiency. This algorithm has three main differences from
    the Ant System algorithm:

    1.  A different transition probability function which provides a direct way to balance 
        between exploration of new edges, and exploitation of a priori and accumulated knowledge.

    2.  The global pheromone update rule is applied only to the edges of the best tour 
        found so far.

    3.  A local pheromone update rule is applied to all edges while ants build a solution.
    """
    def __init__(
        self, 
        *args,
        rho=0.1, # pheromone decay in local update
        q0=0.9, # probability of choosing the greedy strategy
        gamma=0.99, # gamma value for Ant-Q
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.rho = rho
        self.q0 = q0
        self.gamma = gamma

        self.best_tour = None
        self.min_cost = np.inf

        self.delta_phero = {
            'ACS': lambda _, __: self.initial_phero_level,
            'Ant-Q': lambda edge, mask: 0 if np.all(~mask) else self.gamma * np.max(self.pheromones[edge[0]][mask]),    
            'zero': lambda _, __: 0
        }
    
    def sample(
        self,
        n_tours=None,
        strategy='stochastic',
        apply_local_pheromone_update=True,
        local_update_strategy='ACS'
        ):
        """
        Sample n_tours from the TSP instance and, if apply_local_pheromone_update is True, apply the local pheromone update rule after each
        step of the ant. If the strategy parameter is set to 'greedy', the ant will always choose the city with the highest probability 
        and the local pheromone update won't be applied.

        """
        assert n_tours is None or n_tours <= self.n_ants, 'Sampling for n_tours > n_ants is not supported'
        if n_tours is None:
            n_tours = self.n_ants

        starting_positions = self._initialize_ants(n_tours)
        tours = []
        
        for ant in range(n_tours):
            initial_position = starting_positions[ant]
            tour = [initial_position]
            default_probabilities = self._transition_probability(initial_position, visited_cities=tour)

            for i in range(self.n_cities-1): 
                if np.random.rand() < self.q0 or strategy == 'greedy': 
                    # exploit: greedily choose the next city
                    next_city = np.argmax(default_probabilities)
                else: 
                    # explore: stochastically sample the next city 
                    next_city = np.random.choice(np.arange(self.n_cities), p=default_probabilities)
                    
                tour.append(next_city)  
                default_probabilities[next_city] = 0.0
                default_probabilities = default_probabilities / default_probabilities.sum() # normalizing again after removing the probability of the chosen city

                if apply_local_pheromone_update and strategy != 'greedy':
                    visited_cities = np.zeros(self.n_cities, dtype=bool)
                    visited_cities[tour] = True
                    self.local_update_pheromone((tour[-2], tour[-1]), strategy=local_update_strategy, visited_cities=visited_cities)

            tours.append(tour)

        return np.array(tours)
    
    def get_best_tour(self, tours):
        costs = self.instance.evaluate(tours)
        if np.min(costs) > self.min_cost:
            return self.best_tour        
        
        self.min_cost = np.min(costs)
        self.best_tour = tours[np.argmin(costs)]
        return self.best_tour
    
    def update_pheromone(self, tours, check_best_tour=True):
        best_tour = self.get_best_tour(tours) if check_best_tour else self.best_tour

        # the pheromone matrix is updated only for the best tour found so far
        delta_pheros = np.zeros((self.n_cities, self.n_cities))
        tour = np.concatenate((best_tour, [best_tour[0]])) # the ant returns to the starting city
        edges = np.array([[city, next_city] for city, next_city in zip(tour[:-1], tour[1:])])

        # the i-th element of the array below is the city visited after city i
        next_cities = edges[edges[:, 0].argsort()][:, 1]

        # scattering ones in the positions that correspond to edges walked by the ant
        values = np.ones(delta_pheros.shape[1])
        values[np.where(next_cities == tour[0])[0]] = 0 # discarding the edge which closes the loop
        np.put_along_axis(delta_pheros, np.expand_dims(next_cities, 1), np.expand_dims(values, 1), axis=-1)
        delta_pheros = (delta_pheros + delta_pheros.T) / self.min_cost # making the matrix symmetric and normalizing by the tour length

        # applying the pheromone update rule 
        self.pheromones = (1 - self.alpha) * self.pheromones + self.alpha * delta_pheros

    def local_update_pheromone(self, edge: Tuple[int, int], strategy: str='ACS', visited_cities=None):
        if strategy == 'off':
            return
        assert strategy in ['ACS', 'Ant-Q', 'zero']

        self.pheromones[edge] = (1 - self.rho) * self.pheromones[edge] + self.delta_phero[strategy](edge, ~visited_cities)
        self.pheromones[edge[::-1]] = self.pheromones[edge]
        
        
        

        