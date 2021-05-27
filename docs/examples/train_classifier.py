import os

import nanotune as nt
from nanotune.classification.classifier import Classifier

nt_path = os.path.dirname(os.path.dirname(os.path.abspath(nt.__file__)))

classifier = Classifier(
    ['pinchoff.npy'],
    'pinchoff',
    data_types=["signal"],
    classifier="SVC",
    folder_path=os.path.join(nt_path, 'data', 'training_data'),
)

classifier.train()
