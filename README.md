# DeepCryoRNA - deep learning-based RNA structure reconstruction from cryo-EM maps


## Platform Requirements (Tested)
The following are tested system settings:

* GNU/Linux x86_64 (Ubuntu 16)
* gcc/g++ supporting C++11 (>= version 4.7)


## Installation

#### 1. Install Chimerax Release 1.3
Please download and install Chimerax Release 1.3 from https://www.rbvi.ucsf.edu/chimerax/older_releases.html.

Please check if Chimerax is installed:
```
chimerax --version
```

#### 2. Install Anaconda with Python >= version 3.10
Please download and install Anaconda from https://www.anaconda.com/download#downloads.

#### 3. Install TensorFlow 2
Please download and install Tensorflow from https://www.tensorflow.org/install/pip.

Please install TensorFlow with the *pip* package manager in Python from the newly installed Anaconda software.

Please follow the installation instructions for TensorFlow to install the NVIDIA software for GPU support. If not installed properly, only CPUs are used to predict atom classes by the neural network, which will slow down the prediction process. 

#### 4. Clone this repository on your local machine
```
git clone https://github.com/Vfold-RNA/DeepCryoRNA.git ${HOME}/DeepCryoRNA
```
```
wget https://github.com/Vfold-RNA/DeepCryoRNA/releases/download/v1.0/DeepCryoRNA_Unet.hdf5 -P ${HOME}/DeepCryoRNA/src
```

#### 5. Install the required Python packages, compile QRNAS <sup>[1]</sup> for energy minimization, and generate shared C++ library for efficient calculation of global sequence alignment scores
```
cd ${HOME}/DeepCryoRNA
```
```
bash ./install.sh
```

#### 6. Set the environment variable *DEEPCRYORNA_HOME*
Add the following line to ${HOME}/.bashrc:
```
export DEEPCRYORNA_HOME="${HOME}/DeepCryoRNA/src"
```

Open a new terminal and check the environment variable *DEEPCRYORNA_HOME*:
```
echo $DEEPCRYORNA_HOME
```

## Usage of DeepCryoRNA

#### Run DeepCryoRNA for the example RNA 6UES
```
cd ${HOME}/DeepCryoRNA/Examples/example_6UES
```
Please change "Path_To_Anaconda" to the real path to the newly installed Anaconda in the following command:
```
Path_To_Anaconda/bin/python ${DEEPCRYORNA_HOME}/main.py -i ./input_6UES.txt > DeepCryoRNA_6UES.log
```
The input file "input_6UES.txt" includes the information for the RNA name, the cryo-EM map, RNA sequence, contour level, gpu and cpu. Please see the README.md in the *Examples/* folder for further information.

The log file *DeepCryoRNA_6UES.log* stores the progress information.

Please see the README.md in the *Examples/* folder for the information regarding the output files and folders.


## Software References

[1] https://genesilico.pl/software/stand-alone/qrnas
