# Deep Adjoint Matching Neural Networks
The objective is to train a neural network as the surrogate to the physical model that can make predictions one step at a time through time. Unlike a standard neural network training process (usually calculating the loss with the forward prediction and minimizing the loss), an adjoint matching neural network is additionally trained to match the adjoint of the forward model so that the surrogate model can return accurate sensitivity for its inputs. 

Adjoint matching neural network has the following general produce.
1. Build the neural network model, $\mathcal{N}_{\theta}$ , parametrizd by $\theta$, that approximates the true physical model, $\mathcal{M}$. The physical forward model, starting with the initial condition, pushes the current solution to the next time step. $$ \mathcal{N}_{\theta} \approx \mathcal{M}$$, where $$x_{t+1} = \mathcal{M}(x_{t}, m)$$ Here $m$ are additional parameters that affect the solution $x$ but not considered part of the solution at time $t$. 
2. Calculate the neural network surrogate Jacobian with automatic differentiation. $$N = \frac{\mathcal{\partial N}}{\partial \mathbf{x}},$$ where $\mathbf{x} = (x, m)$, and $N \in \mathbb{R}^{n \times p}$, $n$ being the dimension of the solution and $p$ the dimension of the input. Now the model adjoint is the transpose of the Jacobian, $N^T$. 
3. Train the neural network by minimizing the loss function $L_{tot}$ via gradient-based optimization approaches. More specifically, the loss function has the following form. $$ L_{tot} = L_{fwd} + \alpha L_{adj},$$ where $L_{fwd}$ is the loss term for the forward predictions, $L_{adj}$ is the loss term for the model adjoints, and $\alpha$ controls the relative importance of the adjoint loss term in the total loss. 

### How to cite
```
@misc{Deepadjoint2023,
  author = {Yixuan Sun},
  title = {Deep Adjoint Matching Neural Networks},
  howpublished = {GitHub repository},
  year = {2023},
  month = {March},
  version = {v1.0},
  url = {https://github.com/iamyixuan/DeepAdjoint}
}
```
