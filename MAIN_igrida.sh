#!/bin/bash

# SCRIPT dir
SCRIPTDIR=/udd/aackaouy/OT-DA
echo "$SCRIPTDIR"

chmod u+x $SCRIPTDIR/bypass.sh

oarsub -I -l {"host = 'igrida-abacus4.irisa.fr'"}/nodes=1/gpu_device=1,walltime=1:00:0
set -xv

. /etc/profile.d/modules.sh

module load cuDNN/v7.0.4
module load cuda/9.0.176

# Activate the py virtual environnement
. /udd/aackaouy/myVE/bin/activate
PYTHONHASHSEED=0 python3 -W ignore $SCRIPTDIR/main.py

#oarsub -t besteffort -t idempotent -p "dedicated='none' or dedicated = 'serpico' " -l {"gpu_model = 'Tesla P100'"}/gpu_device=1,walltime=48:00:0 "$SCRIPTDIR/bypass.sh"

#oarsub -t besteffort -t idempotent -p "dedicated='none' or dedicated = 'serpico'" -l  {"host = 'igrida-abacus.irisa.fr'"}/gpu_device=1,walltime=48:0:0 "$SCRIPTDIR/bypass.sh" 

#oarsub -l {"gpu_model = 'Tesla P100'"}/gpu_device=1,walltime=10:00:0 "$SCRIPTDIR/bypass_valverde_FG.sh"


