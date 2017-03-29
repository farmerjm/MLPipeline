'''
Load different data set types
'''

def load_fits_images(filenames):
    '''Expects filenames to be a list of .fits file locations'''
    from astropy.io.fits import getdata
    return [ getdata(filename).copy() for filename in filenames ]


