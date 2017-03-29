import sys
from sklearn.cross_validation import train_test_split
from MLPipeline.IO.config_parser import parse_configfile
from MLPipeline.IO.collect_classes import get_two_classes
from _tools import build_parameter_grid, grid_search
from MLPipeline.pipeline_tools.build_pipeline import build_pipeline
import time

cfg = parse_configfile(sys.argv[1])
start_time = time.time()

# Collect data and labels
X, y = get_two_classes(cfg['filenames']['non_lens_glob'], 
                       cfg['filenames']['lens_glob'])

assert( len(X) == len(y) )
assert( len(X) > 0 )

# Split the dataset and labels into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2 )

assert( len(X_train) == len(y_train) )
assert( len(X_test) == len(y_test) )
assert( len(X_train) > 0 )
assert( len(X_test) > 0 )

# Build the parameter grid
param_grid = build_parameter_grid(cfg['param_grid'])

# Build the pipeline
pipeline = build_pipeline(cfg['image_processing'].values(), 
                          cfg['classifier']['label'])

# Perform the grid search
grid_search(pipeline, param_grid, X_train, y_train, X_test, y_test) 

print 'Time Taken:', time.time()-start_time
