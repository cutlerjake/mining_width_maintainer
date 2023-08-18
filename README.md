# mining_width_maintainer
Perturb open pit mine schedules, while maintaining desired mining width.

Open pit mine production schedules are required to satisfy a mining width constraint, which ensures feasible equipment access to scheduled mining activities. Open pit mine production schedule optimizers typically resort to block aggregation or Lagrangian relaxation to satify these constraints. However, block aggregation artifically decreases selectivity, negatively effecting profitability, and Langrangian relaxation does not explicity gaurantee satifaction of the mining width constraint.

This library provides a new approach to enable open pit mine production schedule optimizers to satisfy mining width constraints. The library provides MiningWidthMaintainer which allows perturbations through .perturbation(SquareMiningWidth, period). The perturbation is applied by setting all internal inds of the SquareMiningWidth to the specified period, then, all neighboring blocks whose mining width could be violated by the perturbation are checked. Each violation is fixed by finding the SquareMiningWidth that covers the violating index with the smallest change to the current solution, setting all internal inds of SquareMiningWidth to the specified period, and adding its neighbors to the list of neighbors to check for violations.

