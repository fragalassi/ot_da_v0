#!/usr/bin/env bash

mkdir -p Logs
python3 -u main.py -source "08" -target "01" -alpha 1e-6 -jdot "True" -rev 0 > Logs/output_0.log
python3 -u main.py -source "08" -target "08" -alpha 1e-6 -jdot "False" -rev 1 > Logs/output_1.log
python3 -u main.py -source "07" -target "01" -alpha 1e-6 -jdot "False" -rev 2 > Logs/output_2.log

