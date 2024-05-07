### Setup

1. Install RASPy (https://github.com/srush/RASPy) and Tracr (https://github.com/google-deepmind/tracr). Make sure to replace the _rasp.py_ file in RASPy with the _rasp.py_ file in this directory before installing. This mods RASPy to allow for mean aggregation as in the original RASP paper.

2. Pray it works

#### General Notes
Tracr default to categorical and struggles with converting between categorical and numerical. A categorical value can be treated as a number e.g. the indices but it still cannot directly interact with a numerical value. I am not sure what exactly is happening when you treat a categorical number as a number i.e. dividing to categorical numbers and if that will give correct results or strange approximations (similar to what happens when trying to Map numerical to categorical)

Most of my algorithmic work is done with pen and paper instead of blindly trying to test in RASPy/Tracr