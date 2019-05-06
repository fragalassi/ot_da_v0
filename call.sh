
#!/usr/bin/env bash

mkdir -p Logs
python3 -u main.py -source "01" -target "07" -alpha 1e-1 -epochs 1000 -callback "True" -jdot "True" -augment "False" -shape 16 -rev 0 > Logs/output_0.log
