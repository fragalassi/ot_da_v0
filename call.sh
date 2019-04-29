#!/usr/bin/env bash

mkdir -p Logs
python3 -u main.py -source "01" -target "07" -alpha 1e-6 -jdot "False" -rev 0 > Logs/output_0.log

