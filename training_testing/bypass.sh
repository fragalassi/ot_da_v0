#!/bin/bash

SCRIPTDIR=/temp_dd/igrida-fs1/fgalassi/3DUnetCNN-master/brats
echo "$SCRIPTDIR"

. /etc/profile.d/modules.sh
set -xv

#module load nibabel
#module load cuDNN/v6.0  
#module load cuda/8.0.27
#module load tensorfLow/1.3.0-py2.7 
#module load keras

module load cuDNN/v7.0.4
module load cuda/9.0.176

# Activate the py virtual environnement
. /udd/fgalassi/myVE/bin/activate
PYTHONHASHSEED=0 python3 $SCRIPTDIR/train_isensee2017.py
#PYTHONHASHSEED=0 python3 $SCRIPTDIR/evaluate.py
#PYTHONHASHSEED=0 python3 $SCRIPTDIR/predict.py
#PYTHONHASHSEED=0 python3 $SCRIPTDIR/testing_data_FG.py

## Activate the py virtual environnement
#source $SCRIPTDIR/my_env/bin/activate
#python $SCRIPTDIR/train_from_scratch.py

