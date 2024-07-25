Added change log for specifying what has been added or changed when updating the gitHub directory

Found a bug in tracr_new.ipynb where the np.unique function sorts the values in ascending order which 
causes the training data and test data to be unevenly distrubuted. Fixed this in the Model class 
implementation but left tracr_new.ipynb the same for now.

Changes to the Model class
    Added a setWeights method to the Model class
