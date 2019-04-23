#!/bin/bash
set -xv

SCRIPTDIR=/udd/aackaouy/OT-DA
echo "$SCRIPTDIR"

. /etc/profile.d/modules.sh

#module load nibabel
#module load cuDNN/v6.0  
#module load cuda/8.0.27
#module load tensorfLow/1.3.0-py2.7 
#module load keras

module load cuDNN/v7.0.4
module load cuda/9.0.176

# Activate the py virtual environnement
. /udd/aackaouy/myVE/bin/activate

PYTHONHASHSEED=0 python3 -W ignore $SCRIPTDIR/main.py


