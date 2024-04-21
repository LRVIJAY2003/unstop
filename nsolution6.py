import numpy as np

states = ['Beginner', 'Intermediate', 'Advanced']

actions = ['Course A', 'Course B', 'Course C']

rewards = {
    'Beginner': {'Course A': 10, 'Course B': 5, 'Course C': 2},
    'Intermediate': {'Course A': 8, 'Course B': 7, 'Course C': 6},
    'Advanced': {'Course A': 6, 'Course B': 8, 'Course C': 10}
}

q_table = np.zeros((len(states), len(actions)))

alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration-exploitation trade-off

def q_learning(state, q_table):
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.choice(actions)  # Exploration
    else:
        action = actions[np.argmax(q_table[state])]
    
    next_state = np.random.choice(states)  # Simulate next state transition
    reward = rewards[state][action]

    q_table[state, actions.index(action)] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, actions.index(action)])
    
    return next_state

population_size = 10
num_generations = 100

def genetic_algorithm():
    population = np.random.choice(actions, size=(population_size, len(states)))
    
    for generation in range(num_generations):
        fitness_scores = [fitness(path) for path in population]
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Sort in descending order

        parents = population[sorted_indices[:population_size // 2]]

        offspring = []
        for _ in range(population_size // 2):
            parent1, parent2 = np.random.choice(parents, size=2, replace=False)
            crossover_point = np.random.randint(1, len(states))
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            mutation_prob = 0.1
            if np.random.uniform(0, 1) < mutation_prob:
                mutation_index = np.random.randint(len(states))
                child[mutation_index] = np.random.choice(actions)
            offspring.append(child)
        
        # Update population with offspring
        population = np.vstack((parents, np.array(offspring)))
    
    best_path = population[np.argmax([fitness(path) for path in population])]
    return best_path

# Fitness function (e.g., based on relevance, coherence, balance)
def fitness(path):
    return sum(rewards[state][action] for state, action in zip(states, path))

# Main function
def main():
  
    state = 'Beginner'
    
    for _ in range(100):
        state = q_learning(states.index(state), q_table)
    
    best_path = genetic_algorithm()
    print("Best learning path found by genetic algorithm:", best_path)

if __name__ == "__main__":
    main()
