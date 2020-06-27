## Real-time dynamic programming

Realtime dynamic programming (RTDP) samples paths through the state space based on the current greedy policy and updates the values along its way.
It's an efficient way of real-time planning, since not necessarily the whole state space is visited, and works well for stochastic environments.
This implementation assumes a full observability.

One such stochastic environment is the frozen lake environment.
In this repository RTDP is applied to a 20x20 map of said environment.
After training a few minutes, an average reward of 0.48 over 10000 evaluations can be observed.

For more information on RTDP, refer to:
* Learning to act using real-time dynamic programming, 1994, Barto et al
* Planning with Markov Decision Processes: An AI Perspective, 2012, Mausam and Kobolov
