implementing HamiltonianFlow with simple 2D dataset

accepting concept from Neural ODE
paper link: https://arxiv.org/abs/1806.07366

For the stablity and memory efficiency, I tried to apply checkpoint backpropagation instead of applying adjoint method
relative method is in the link below
https://arxiv.org/abs/1902.10298


as a result, the performance was not as decent as any other flow-based model(Glow, FFJORD) but it was worth trying in that
1. realizing adjoint method has a few drawbacks including computing inefficiency, instability
2. performance of neural ODE is no better than conventional method


