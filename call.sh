python3 -u main.py -source '01' -target '07' -split_list '(([0, 2, 1, 4], [3]), ([0, 3, 2, 4], [1]))' -intensity_ceil -0.40 -alpha 40 -beta 10 -jdot 'False' -epochs 60 -load_model 'True' -rev 0 > Logs/output_0.log
python3 -u main.py -source '01' -target '08' -split_list '(([0, 2, 1, 4], [3]), ([0, 1, 2, 4], [3]))' -intensity_ceil -0.40 -alpha 40 -beta 10 -jdot 'False' -epochs 60 -load_model 'True' -rev 0 > Logs/output_0.log
python3 -u main.py -source '08' -target '01' -split_list '(([0, 2, 1, 4], [3]), ([0, 1, 2, 4], [3]))' -intensity_ceil -0.40 -alpha 40 -beta 10 -jdot 'False' -epochs 60 -load_model 'True' -rev 0 > Logs/output_0.log
