# Frozen-Lake-OpenAI-PSO-GA


Frozen Lake from Open AI Gym is a strategy game that involves crossing a frozen lake from Start(S) to Goal(G) without falling into any Holes(H) by walking over the Frozen(F) lake. The agent may not always move in the intended direction due to the slippery nature of the frozen lake.

Link to Open source - https://www.gymlibrary.dev/environments/toy_text/frozen_lake/

An experiment on a discrete game has never been done using these algorithms. In this section, we explore and visualize the experiment done on a game by Open AI Gym. This experiment shows a clear indication of whether these algorithms have the requirements to solve a discrete puzzle game.

The outcome of the experiment has provided some insights into how PSO and Genetic Algorithms can be used to optimise path-finding for the agent. The initial step to optimize both PSO and GA was to check if the particle’s best position would be closer the the global optima which is D - 1(D represent the dimensionality of the search space). GA tends to average positional score of 7.1 which is not very optimal whereas PSO has a downward trajectory for its overall fitness score. Throughout the iteration, the particles tend to move towards the reward which is at the end position of the map. The algorithm has been modified to fulfil the agent’s goal which is to reach the reward as much as possible by taking fewer steps over time. Compared to GA, PSO performs quite well in navigating the agent to the reward. GA tends to sway away from the reward and explore other options thus increasing the steps taken in each iteration. However, there is still a huge problem with mitigating the agent after reaching the goal of falling less into the hole and reaching the goal. The problem lies in the code itself as there is no clear indication to store the agent’s historical movements. However, it can be achieved by adding neural network to store the data of the agent’s historical movements.

The project shows the possibility of swaying from the traditional ap- proach to a dynamic approach to solve path-finding in computational problems using Swarm Intelligence. The project shows a clear indi- cation that algorithms like PSO and GA can be used to optimize and solve path-finding problems. The results might not be as straightfor- ward as other path-finding algorithms such as Breadth First Search or Dijkstra’s algorithm but they can be used in 3d games in particular where the dimensional scope of the problem is much higher and can show great results. The future improvements for this project would be to fine-tune the parameters for both PSO and GA to achieve the desir- able results. Some of the changes can be made to the PSO algorithm by changing the values of inertia weight, social weight and cogni- tive weight to get the best possible results. Another change could be made to implement policies to the agent, restricting certain move- ments from the agent might be a feasible solution. For GA, instead of using traditional single-point crossover using two-point crossover and uniform crossover may produce better results.


Key Features:
<ul>
<li>Utilizes Particle Swarm Optimization and Genetic Algorithm for optimization.</li>
<li>Integrates with the FrozenLake-v1 environment from OpenAI Gym.</li>
<li>Includes performance evaluation using a custom fitness function based on the agent's rewards, falls, and steps.</li>
</ul>
Future Improvements:
<ul>
<li>Algorithm Performance Tuning: Experiment with different optimization parameters to improve convergence speed and solution accuracy.</li>
<li>Multi-agent Optimization: Extend the project to include multi-agent optimization approaches.</li>
<li>Larger Environment Sizes: Scale the project to work with larger and more complex environments.</li>
<li>Comparison with Other Algorithms: Implement additional optimization algorithms like Q-learning or Deep Q Networks (DQN) for benchmarking.</li>
<li>Visualization Enhancements: Add more detailed visualizations to track agent performance over time.</li>
</ul>
