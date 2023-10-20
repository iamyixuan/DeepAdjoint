# Deep Adjoint Matching Neural Networks
The objective is to train a neural network as the surrogate to the physical model that can make predictions one step at a time through time. Unlike a standard neural network training process (usually calculating the loss with the forward prediction and minimizing the loss), an adjoint matching neural network is additionally trained to match the adjoint of the forward model so that the surrogate model can return accurate sensitivity for its inputs. 

Adjoint matching neural network has the following general produce.
1. Build the neural network model, $\mathcal{N}_{\theta}$ , parametrizd by $\theta$, that approximates the true physical model, $\mathcal{M}$. The physical forward model, starting with the initial condition, pushes the current solution to the next time step. $$ \mathcal{N}_{\theta} \approx \mathcal{M}$$, where $$x_{t+1} = \mathcal{M}(x_{t}, m)$$ Here $m$ are additional parameters that affect the solution $x$ but not considered part of the solution at time $t$. 
2. Calculate the neural network surrogate Jacobian with automatic differentiation. $$N = \frac{\mathcal{\partial N}}{\partial \mathbf{x}},$$ where $\mathbf{x} = (x, m)$, and $N \in \mathbb{R}^{n \times p}$, $n$ being the dimension of the solution and $p$ the dimension of the input. Now the model adjoint is the transpose of the Jacobian, $N^T$. 
3. Train the neural network by minimizing the loss function $L_{tot}$ via gradient-based optimization approaches. More specifically, the loss function has the following form. $$ L_{tot} = L_{fwd} + \alpha L_{adj},$$ where $L_{fwd}$ is the loss term for the forward predictions, $L_{adj}$ is the loss term for the model adjoints, and $\alpha$ controls the relative importance of the adjoint loss term in the total loss. 

# Train Neural Forward Surrogates
To train neural networks for modeling the forward dynamics in a distributed way, one can directly excuate `SOMAforward.py` using `Slurm` job script, as follows.
```
#!/bin/bash
#SBATCH --output=./_job_history/%j.out
#SBATCH --error=./_job_history/%j.err
#SBATCH -A m4259
#SBATCH --qos=regular
#SBATCH --nodes=10
#SBATCH --constraint=gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time 12:00:00 


# conda activate deeplearning
export WORLD_SIZE=40
export MASTER_PORT=38174
export TRAIN=1  
export LOSS=FNO

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)


export MASTER_ADDR=$master_addr
export LOSS=FNO
srun python SOMAforward.py -batch_size 8\
                      -epochs 2000\
                      -lr 0.0001\
                      -model_name FNO-\
                      -mask False\
                      -net_type FNO\
                      -data GM\

```
There are a few important environment variables. `WORLD_SIZE` specifies the total number of GPUs being used to train the model. `LOSS` specifies the loss function type for training the model. Here we use `FNO` loss, which essentially is the Lp loss of the target values and predicted ones. 

## Make Predictions
It is straightforward to make predictions with the testing set using `SOMAforward.py`. Make sure to set flag `-train False`. This does not require distributed training setup and can be done using a single GPU. 

Example
```
python SOMAforward.py -train False -net_type FNO -model_path PATH/TO/SAVED/MODEL
```

## Perform Rollout
The purpose of performing rollout is to investigate model performance in the autoregressive way, where new predictions are made using the previous predictions as input. Similar to making predictions, this can be done using `SOMAforward.py` with flag `-train rollout`. 

Example
```
python SOMAforward.py -train rollout -net_type FNO -model_path PATH/TO/SAVED/MODEL
```


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


