
#!/usr/bin/env bash

mkdir -p Logs
python3 -u main.py -source "01" -target "07" -alpha 1e-1 -epochs 1000 -jdot "True" -augment "False" -shape 16 -rev 0 > Logs/output_0.log
python3 -u main.py -source "01" -target "07" -alpha 1e-1 -epochs 500 -jdot "True" -augment "False" -shape 16 -rev 1 > Logs/output_1.log
python3 -u main.py -source "01" -target "07" -alpha 1e-1 -epochs 200 -jdot "True" -augment "False" -shape 16 -rev 2 > Logs/output_2.log

python3 -u main.py -source "01" -target "07" -alpha 1 -epochs 1000 -jdot "True" -augment "False" -shape 16 -rev 3 > Logs/output_3.log
python3 -u main.py -source "01" -target "07" -alpha 5 -epochs 1000 -jdot "True" -augment "False" -shape 16 -rev 4 > Logs/output_4.log
python3 -u main.py -source "01" -target "07" -alpha 10 -epochs 1000 -jdot "True" -augment "False" -shape 16 -rev 5 > Logs/output_5.log
python3 -u main.py -source "01" -target "07" -alpha 100 -epochs 1000 -jdot "True" -augment "False" -shape 16 -rev 6 > Logs/output_6.log

python3 -u main.py -source "01" -target "07" -alpha 1e-1 -epochs 200 -jdot "True" -augment "True" -shape 16 -rev 7 > Logs/output_7.log
