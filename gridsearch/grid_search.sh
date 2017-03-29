# Use this bash script to run a grid search on config parameters
# defined in a directory in $1 (first command line argument
# Prints stdout and stderr to the grid_search.out file

python model_grid_search.py $1 >& $1/grid_search.out