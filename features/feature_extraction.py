from sklearn.base import BaseEstimator
import numpy as np


''' This contains a collection of feature extraction classes which can
be used in the Pipeline and GridSearch.  

To create another class, you need to have a fit, transform, and
fit_transform method.  The fit method must return itself, and the
fit_transform method must return the self.transform method for all X.

Add this class to the dictionary with an easy-to-remember string for
its key.
'''

class HOG(BaseEstimator):
    def __init__( self, orientations = 9, pixels_per_cell = (8, 8),
                 cells_per_block = (3, 3) ):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def fit(self, images, y = None):
        return self

    def transform(self, images):
        from skimage.feature import hog
        return np.array([hog(image, orientations = self.orientations,
                             pixels_per_cell = self.pixels_per_cell,
                             cells_per_block = self.cells_per_block,
                             )
                         for image in images])

    def fit_transform(self, images, y = None):
        return self.transform(images)


feature_extraction_opts = {
    'hog' : HOG(),
    }
