
#!/usr/bin/env bash

mkdir -p Logs
python3 -u main.py -source "All" -target "07" -alpha 1e-6 -jdot "False" -augment "True" -shape 16 -rev 0 > Logs/output_0.log
python3 -u main.py -source "All" -target "07" -alpha 1e-6 -jdot "False" -augment "False" -shape 8 -rev 1 > Logs/output_1.log
python3 -u main.py -source "All" -target "07" -alpha 1e-6 -jdot "False" -augment "False" -shape 16 -rev 2 > Logs/output_2.log
python3 -u main.py -source "All" -target "07" -alpha 1e-6 -jdot "False" -augment "False" -shape 32 -rev 3 > Logs/output_3.log
