python3 -u main.py -source '01' -target '07' -intensity_ceil -0.1 -alpha 0.01 -beta 0.001 -jdot 'False' -epochs 2000 -load_model 'False' -rev 0 > Logs/output_0.log
python3 -u main.py -source '07' -target '08' -intensity_ceil -0.1 -alpha 0.01 -beta 0.001 -jdot 'False' -epochs 2000 -load_model 'False' -rev 1 > Logs/output_1.log
python3 -u main.py -source '08' -target '01' -intensity_ceil -0.1 -alpha 0.01 -beta 0.001 -jdot 'False' -epochs 2000 -load_model 'False' -rev 2 > Logs/output_2.log
