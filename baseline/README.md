# baseline code source
'Fair_SC_normalized.m' is from the [MATLAB source code](https://github.com/matthklein/fair_spectral_clustering/blob/master/Fair_SC_normalized.m) of [BSE](http://proceedings.mlr.press/v97/kleindessner19b/kleindessner19b.pdf). We discard the Kmeans phase(Line 61) for ablation study.

'cplex_fair_assignment_lp_solver.py', 'cplex_violating_clustering_lp_solver.py', 
'iterative_rounding.py' and 'util' are from the [source code](https://github.com/nicolasjulioflores/fair_algorithms_for_clustering) of [fair k-means](https://proceedings.neurips.cc/paper/2019/file/fc192b0c0d270dbf41870a63a8c76c2f-Paper.pdf).

'fairwalk.py' is based on the [source code](https://publications.cispa.saarland/2933/1/IJCAI19.pdf) of [Fairwalk](https://github.com/urielsinger/fairwalk).The whole project is merged into one .py file for ease of use.