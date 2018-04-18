
# academic-projects
bioinformat
## Background
The code was written to be trained using the COMAV private data set of flowers, fruits, trichomas and leafs, but it can be easily modified to be used in other 2D applications.

## Training
'''1. Install dependencies:
'''

keras,

tensorflow,

pillow
'''

2. Place the data folders in the pathwork/DATA folder.

3. Run the training: Part 1 - Part 2.
'''

$ python cnn_bioinf.py

## Writing prediction labels from predict set
Predicted labels and the values of accucary, recall (sensibity) and predict (specificity) of the predict dataset achieved by the classifier will be returned.

'''1. Place the data set in the path_work/PREDICTION_DATA folder and its groundtruth labels for te metrics results.

2. Run the predict: Part 3.
'''

$ python cnn_bioinf_predict.py
