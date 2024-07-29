import numpy as np
from tsp import TSP
from nearestneighbor import NearestNeighborSolver

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
        
        

        