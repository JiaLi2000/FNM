# Fair Ncut Minimization

## Requirements
Our experiments test on ubuntu 20.04 with python 3.9 and [Gurobi 10.0](https://www.gurobi.com/).
Dependent python libraries of our algorithm **FNM** include: \
numpy==1.21.5\
pandas==1.3.5\
scipy==1.9.1\
networkx==2.6.3\
scikit-learn==1.1.1\
To run other baseline algorithms,
IP/LP sovler [Cplex](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-optimizer), [MATLAB](https://ww2.mathworks.cn/products/matlab.html) and python library [karateclub=1.3.3](https://karateclub.readthedocs.io/en/latest/) are also needed.
Please refer to the corresponding paper for details.

## Reproduction of results
1. Create a new directory 'results/embeddings' in the root directory.
2. Run 'run_exp_FESC.m' with MATLAB.
3. Run any of the other experimental scripts simply by
```python scripts_name.py```.