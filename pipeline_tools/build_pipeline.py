from sklearn.pipeline import Pipeline
from MLPipeline.features.feature_extraction import feature_extraction_opts
from MLPipeline.features.data_preprocessing import preprocessing_opts
from MLPipeline.ml.classifiers import classifiers

def build_pipeline(preprocessing_labels, feature_extraction_label, classifier_label) :
    '''
    |
    |   Create the pipeline which consists of 
    |   pre-processing step(s), feature extraction,  and a classifier
    |
    '''

    estimators = []


    for label in image_processor_labels :
        estimators.append((label, preprocessing_opts[label]))

    estimators.append((feature_extraction_label, feature_extraction_opts[feature_extraction_label]))

    estimators.append((classifier_label, classifiers[classifier_label]))

    return Pipeline(estimators)

