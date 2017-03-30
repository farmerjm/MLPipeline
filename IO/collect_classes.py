import glob
from loaders import load_fits_images

def get_two_classes( label_glob1, label_glob2 ) :
    ''' Returns the a list of images and a list of the labels '''
    label_filenames1 = glob.glob(label_glob1)
    label_filenames2 = glob.glob(label_glob2)
    all_filenames = label_filenames1 + label_filenames2

    return load_fits_images(all_filenames), [0] * len(label_filenames1) + [1] * len(label_filenames2)

def get_class_labels( *class_globs ) :
    '''Exercise: Write a function for a general number of classes to classify '''

    pass
    
