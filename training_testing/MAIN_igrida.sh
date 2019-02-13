#!/bin/bash

# SCRIPT dir
SCRIPTDIR=/temp_dd/igrida-fs1/fgalassi/3DUnetCNN-master/brats
echo "$SCRIPTDIR"

chmod +x $SCRIPTDIR/bypass.sh

#oarsub -l {"host = 'igrida-abacus4.irisa.fr'"}/nodes=1/gpu_device=1,walltime=48:00:0 "$SCRIPTDIR/bypass.sh"

oarsub -t besteffort -t idempotent -p "dedicated='none' or dedicated = 'serpico' " -l {"gpu_model = 'Tesla P100'"}/gpu_device=1,walltime=48:00:0 "$SCRIPTDIR/bypass.sh"

#oarsub -t besteffort -t idempotent -p "dedicated='none' or dedicated = 'serpico'" -l  {"host = 'igrida-abacus.irisa.fr'"}/gpu_device=1,walltime=48:0:0 "$SCRIPTDIR/bypass.sh" 

#oarsub -l {"gpu_model = 'Tesla P100'"}/gpu_device=1,walltime=10:00:0 "$SCRIPTDIR/bypass_valverde_FG.sh"


