This project uses simulated annealing to find the optimal path for performing a DMRG calculation of a model on the 2d square lattice.

The SA_HP_v*.cpp is the main code in C++ which needs to be compiled linking to OpenMP for parallelization.
It produces the file path.csv in which the vertices are numbered according to the optimal Hamiltonian path/Optimal Layout.

The utility vis_path_C_v*.py visualizes the path using MathPlotLib

The script HP_DMRG_EXT_Corr.py uses the path in path.csv to find the GS of a Heisenberg \pm J spin glass on the square lattice. 
It uses TenPy for the DMRG and it is fully customizable in terms of the bond dimension. It reads the system size from path.csv

It outputs a file Correlations_*.csv with all the <S_i\cdot S_j> correlations expectation values.

The script Compute_Q.py uses the output Correlations_*.csv to compute the value of the Edwards-Anderson order parameter.
