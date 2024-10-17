
#References
#https://machinelearningmastery.com/a-gentle-introduction-to-particle-swarm-optimization/
#https://nathanrooy.github.io/posts/2016-08-17/simple-particle-swarm-optimization-with-python/
#https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
#https://pypi.org/project/geneticalgorithm/
#https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6

import gym
import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt

np.random.seed(42)

env_size = 4  # Set the size of the map
env = gym.make('FrozenLake-v1', desc=generate_random_map(size=env_size), is_slippery=True, render_mode='rgb_array')

def fitness_function(X):
    total_reward = 0
    total_falls = 0  
    total_steps = 0
    for _ in range(maxIteration):
        state = env.reset()[0]
        curr_reward = 0  # reward when the agent reaches the goal counter
        done = False  # checks if the agent reaches the goal
        steps_per_goals = []  # steps taken to reach goal
        fail_list = set()  # check the position every time the agent falls
        
        #main game loop
        while not done:
            action = best_position[np.random.choice(num_particles)][state]  # chooses what action the agent takes Left(0) Right(1) Up(2) or Down(3) based on particle position
            try:
                next_state, reward, done, _ = env.step(int(action))[:4]
                
                # checks if the next state is a hole(H)
                if env.desc.flatten()[next_state] == b'H':
                    reward -= 1
                    total_falls += 1

                    # checks and adds the hole position to the fail_list
                    if next_state not in fail_list:
                        fail_list.add(next_state)

                # check if the agent next step is the goal state
                if next_state == goal_state:
                    reward += 20  # add reward
                    done = True  # reset position if true
                    steps_per_goals.append(env._elapsed_steps)
                    
                curr_reward += reward
                state = next_state
                total_steps += 1
            except KeyError:
                continue

        total_reward += curr_reward
        print("Running through iteration")
        average_steps = total_steps / maxIteration
    return average_steps, total_falls, total_steps, steps_per_goals

# population for Genetic Algorithm and Iteration for PSO
maxIteration = 1000

# particles for PSO
num_particles = 100

# dimension is bound the search space of the game
num_dimensions = env.observation_space.n

# upper bound is the max that the agent can take in the action space
upper_bound = env.action_space.n - 1

# lower bound is [0,0] but set as an int 0.
lower_bound = 0

# function to try to move the particle from the start position to the goal state
def generate_initial_positions(num_particles, num_dimensions, starting_point, goal_position, deviation=1.0):
    initial_positions = np.zeros((num_particles, num_dimensions))
    for i in range(num_particles):
        initial_positions[i] = starting_point + np.random.uniform(-deviation, deviation, size=num_dimensions)
    return initial_positions

deviation = 0.5

# Set the goal position (assuming the goal is at the bottom-right corner)
goal_state = env_size * env_size - 1

# settiing the position of the particles to be at the starting position
position = generate_initial_positions(num_particles, num_dimensions, starting_point=0, goal_position=goal_state, deviation=deviation)

# initial velocity of the particle
velocities = np.random.uniform(lower_bound, upper_bound, size=(num_particles, num_dimensions))

# best position after each iteration. initially set to the initial position
best_position = position.copy()

#stores the best particle score throughout the iteration
best_score = np.full(num_particles, fitness_function(position[0])[0])

# initial global best position
global_best_position = position[0].copy() 

#stores the global best particle score throughout the iteration
global_best_score = fitness_function(position[0])[0]

# holds steps taken each iteration
steps_per_iteration = []

# hold steps taken each time the agent reaches the goal
steps_per_goals = []

#amount of times the agents fall in the hole
total_falls = 0

# total steps throughout the iteration counter
total_steps = 0

# PSO formula
inertia_weight = 0.8
cognitive_weight = 1.5
social_weight = 2.5

best_steps = float("inf")

# PSO initialization
for i in range(num_particles):
    r1 = np.random.random(size=num_dimensions)
    r2 = np.random.random(size=num_dimensions)
    updated_velocities = (inertia_weight * velocities[i]) + \
        (cognitive_weight * r1 * (best_position[i] - position[i])) + \
            (social_weight * r2 * (global_best_position - position[i]))

    velocities[i] = updated_velocities
    position[i] += velocities[i]
    average_steps, falls, steps, steps_reached_goal = fitness_function(position[i])
    
    if average_steps < best_score[i]:
        best_score[i] = average_steps
        best_position[i] = position[i].copy()

    if average_steps < global_best_score:
        global_best_score = average_steps
        global_best_position = position[i].copy()

    if average_steps < best_steps:
        best_steps = average_steps
  
    total_falls += falls
    total_steps += steps
    steps_per_iteration.append(average_steps)
    steps_per_goals.append(steps_reached_goal)

steps_when_reached_goal = [steps for sublist in steps_per_goals for steps in sublist]

#crossover
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

#mutation
def mutate(individual, mutation_rate=0.1):
    mutation_mask = np.random.rand(*individual.shape) < mutation_rate
    mutation_values = np.random.uniform(lower_bound, upper_bound, size=individual.shape)
    individual[mutation_mask] = mutation_values[mutation_mask]
    return individual

mutation_rate = 0.1

total_falls_g = 0
total_steps_g = 0
steps_per_iteration_g = []
steps_per_goals_g = []


fitness_scores = []

#Genetic Algorithm
for i in range(num_particles):
    average_steps, falls, steps, steps_reached_goal = fitness_function(position[i])
    fitness_scores.append(average_steps)
    selected_indices = np.argsort(fitness_scores)[:num_particles]
    selected_population = position[selected_indices]
    global_best_index = np.argmin(fitness_scores)
    
    if fitness_scores[global_best_index] < global_best_score:
        global_best_score = fitness_scores[global_best_index]
        global_best_position = position[global_best_index].copy()   

    total_falls_g += falls
    total_steps_g += steps
    steps_per_iteration_g.append(average_steps)
    steps_per_goals_g.append(steps_reached_goal)

#holds the new population after the iteration    
new_population = []
for i in range(num_particles):
    #single-point crossover
    parent_indices = np.random.choice(len(selected_population), size=2, replace=True)
    parent1, parent2 = selected_population[parent_indices]
    child1, child2 = crossover(parent1, parent2)
    child1 = mutate(child1, mutation_rate)
    child2 = mutate(child2, mutation_rate)
    new_population.extend([child1, child2])

steps_when_reached_goal_g = [steps for sublist in steps_per_goals_g for steps in sublist]

population = np.array(new_population)
best_index_value = np.argmin(steps_per_iteration)
best_index_value_g = np.argmin(steps_per_iteration_g)

print("Best Score:", best_score)
print("Global Best Position:", global_best_position)
print("Global Best steps:", best_steps)

print("Steps to reach the goal PSO: ",steps_when_reached_goal)
print("Steps to reach the goal GA: ", steps_when_reached_goal_g)

print("Worst Index Value:", np.argmax(population))
print("Fitness Score:", fitness_scores)
print("Worst Fitness Score:", np.argmax(fitness_scores))
print("Best Fitness Score:", np.argmin(fitness_scores))


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(steps_per_iteration, label='Best Fitness per Generation')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Best Fitness over Generations')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(global_best_position, color='orange')
plt.xlabel('Generation')
plt.ylabel('Global Best Position')
plt.title('Best Global Position over Generations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(fitness_scores, label='Best Fitness per Generation')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Best Fitness over Generations')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(population, color='orange')
plt.xlabel('Generation')
plt.ylabel('Global Best Position')
plt.title('Best Global Position over Generations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()