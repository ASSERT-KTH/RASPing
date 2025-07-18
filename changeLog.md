#### Gustav changes

Added change log for specifying what has been added or changed when updating the gitHub directory. Append any new additions or changes to better keep track of what has been added

Found a bug in tracr_new.ipynb where the np.unique function sorts the values in ascending order which 
causes the training data and test data to be unevenly distrubuted. Fixed this in the Model class 
implementation but left tracr_new.ipynb the same for now.

Changes to the Model class
* Generally I made the model use the class weights instead of weights added as a parameter. This means that any training done via the class changes the starting weights if further trained. If you want to use initial weights again you must use the resetWeights method.
* Added a setWeights method to the Model class
* Added a random weights method to the model class
* Added the foward pass function to the class 
  * In order to do this I mapped the forward function to a global function which is update every time a class instance needs to use it. This way each instance can keep its own structure while using the same functions.
* Added training function to the class which calls the correct instance of the forward pass function
* Added encoded evaluation method to the class

Added Early stopping to training

Added option to evaluate model on validation data during training
* Added parameter valStep (defaults to 0)
  * If valStep is zero behaves as regular training else evaluates model on valuation data every valStep epochs and returns the training results
* Added a jitted evaluation to speed up computation

Changed how the evaluateEncoded method is computed to slightly speed up computation

Ran tests for over training via the tracr_new_Model_class_implementation notebook

Streamlined process and uploaded results in PerformanceTesting/overTraining.ipynb / PerformanceTesting/savedData/overTraining

