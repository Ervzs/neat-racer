# NEAT Racer

2D cars learn to drive a track, generation by generation, using NEAT (NeuroEvolution of Augmenting Topologies).

![NEAT Racer running](assets/screenshot.png)

## How it works

- 30 cars race at once. Each one is driven by its own small neural network.
- Each car sees the track through 5 radar rays. The ray lengths go into its network.
- The network picks one of 4 actions every frame: steer left, steer right, speed up, slow down.
- A car dies if it leaves the road, or if it barely moves for 3 seconds.
- Fitness is the distance driven. When all cars are dead (or 30 seconds pass), NEAT breeds the next generation from the best drivers.

