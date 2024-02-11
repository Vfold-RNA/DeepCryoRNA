We provide an example for the RNA 6UES.

The folder *example_6UES/* is prepared for the user to test using a GPU and 10 CPUs.

The folder *example_6UES_for_reference/* is the folder after running DeepCryoRNA. We provide it for your reference.

We take the folder *example_6UES_for_reference/* as an example to illustrate the files and subfolders within. 

#### 1. input_6UES.txt
This file contains the input information:
```
rna     6UES          # RNA name
map     emd_20755.map # cryo-EM map name; please download the cryo-EM map from EMDB.
seq     GGUC...CGGGUC # RNA sequence; Please use "-" to connect chains if containing multiple chains.
contour 9             # contour level 
gpu     0             # specify the GPU index if having multiple GPUs on your machine; set to -1 to avoid using GPUs
ncpu    10            # specify the number of CPUs to be used.
```

#### 2. emd_20755.map
It is the cryo-EM map for RNA 6UES downloaded from EMDB (https://www.emdataresource.org/EMD-20755).

#### 3. chimerax.log
It is the log file when preprocessing cryo-EM maps by Chimerax.

#### 4. DeepCryoRNA_6UES.log
It is the log file recording the whole process of DeepCryoRNA.

#### 5. tmp/
This folder stores the files for the processed cryo-EM maps, the predicted and clustered atom information, the predicted nucleotides for patch sizes 64 and 128, and the predicted short chains for patch sizes 64 and 128.

#### 6. ps64/
This folder stores the top 10 predicted RNA structures for patch size 64 and their corresponding energy minimized structures starting with "em-".

The file *best_alignment_results_ps64.txt* in this folder records the top 10 alignments and scores, which correspond to the 10 RNA structures.

The emN.log is the QRNAS log file for energy minimization for the N-th RNA structure.

#### 7. ps128/
The files in this folder are the same as in *ps64/*, but for patch size 128.

#### 8. psBoth/
The files in this folder are the same as in *ps64/*, but for both patch sizes.
