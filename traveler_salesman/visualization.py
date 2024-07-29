import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def plot_solution(nodes, tour=None, cost=None, save_dir='results/tours'):
    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.scatter(nodes[:,0], nodes[:,1])
        
    if tour is None:
        plt.gca().set_aspect('equal', adjustable='box')
        return

    for a, b in zip(tour[:-1], tour[1:]):
        x = nodes[[a, b]].T[0]
        y = nodes[[a, b]].T[1]
        plt.plot(x, y, c='r', zorder=-1)

    ax.set_title(f'Tour length: {cost}')
    ax.set_aspect('equal')

    if save_dir is not None:
        plt.savefig(save_dir)

def plot_pheromone_trails(nodes, pheromones, save_dir='results/pheromones'):
    fig = plt.figure(2)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.scatter(nodes[:,0], nodes[:,1])
    colors = plt.cm.jet(np.linspace(0, 1, 256))

    # normalizing pheromone matrix in the range [0, 255)
    if pheromones.min() == pheromones.max():
        pheromones_alpha = np.zeros(pheromones.shape)
    else:
        pheromones_alpha = (pheromones - pheromones.min()) / (pheromones.max() - pheromones.min())
    pheromones_color = (pheromones_alpha * 255).astype(int)
    
    for i in range(pheromones.shape[0]):
        for j in range(i+1, pheromones.shape[1]):
            x = nodes[[i, j]].T[0]
            y = nodes[[i, j]].T[1]
            edge_pheromone = pheromones_color[i,j]
            ax.plot(x, y, c=colors[edge_pheromone], alpha=pheromones_alpha[i, j])

    norm = mpl.colors.Normalize(vmin=0, vmax=1) 
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('jet', 256), norm=norm) 
    sm.set_array([]) 

    plt.colorbar(sm, ax=ax, label="Pheromone intensity") 
    ax.set_title('Pheromone trails')
    ax.set_aspect('equal', adjustable='box')

    if save_dir is not None:
        plt.savefig(save_dir)

def plot_mean_cost(costs, save_dir=f'results/costs'):
    fig = plt.figure(3)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.plot(costs)
    ax.set_title('Mean cost')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    plt.grid()

    if save_dir is not None:
        plt.savefig(save_dir)
        