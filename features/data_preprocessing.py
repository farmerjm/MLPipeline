from sklearn.base import BaseEstimator
import numpy as np

''' This contains a collection of image processing options which can
be used in the Pipeline and GridSearch.

To create another option, the corresponding class must have a
fit, transform, and fit_transform method.  The fit method must return
itself, and the fit_transform method must return the self.transform
method for all X.

Add this class to the dictionary with an easy-to-remember string for
its key.
'''

class Clip(BaseEstimator) :
    '''Numpy clip to get rid of data regions we don't care about.'''
    def __init__( self, lower=1e-6, upper=1e100):
        self.clip_min = lower
        self.clip_max = upper

    def fit(self, image, y = None) :
        return self

    def transform( self, images ) :
        return np.array( [ np.clip( image, self.clip_min, self.clip_max )
                           for image in images ] )

    def fit_transform( self, images, y = None ) :
        return self.transform(images)

class Log(BaseEstimator) :
    def __init__( self, shift = None ) :
        self.shift = shift

    def _make_positive( self, image ) :
        ''' Ensure that the minimum value is just above zero, or use specified shift'''
        if self.shift == None :
            return image - image.min() + np.abs(image.min())
        else :
            assert( (image+self.shift >= 1.0).all() )
            return image + self.shift

    def fit( self, images, y = None ) :
        return self

    def transform( self, images ) :
        return np.array( [ np.log( self._make_positive(image) ) for image in images ] )

class Rescale(BaseEstimator) :
    def __init__( self, max_value=1. ) :
        self.max_value = max_value

    def fig( self, images, y=None ) :
        return self

    def transform( self, images ) :
        return np.array([ image/image.max() for image in images ])
    

preprocessing_opts = {
    'clip': Clip(),
    'log': Log(),
    'rescale': Rescale(),
    }
