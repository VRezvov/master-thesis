---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

Set the learning rate of each parameter group using a cosine annealing schedule, where $\eta_{max}$ is set to the initial lr, $T_{cur}$ is the number of epochs since the last restart and $T_{i}$ is the number of epochs between two warm restarts in SGDR:

$$
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)
$$

When $T_{cur}=T_{i}$, set $\eta_t = \eta_{min}$.

When $T_{cur}=0$ after restart, set $\eta_t=\eta_{max}$.

It has been proposed in  `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

        Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

```python

```
