# NLP-School-Project

The baseline is adapted from https://github.com/monologg/GoEmotions-pytorch.

To run the current baseline, use the following code in your command line
```
$ python3 run_goemotions.py --taxonomy {$TAXONOMY}

$ python3 run_goemotions.py --taxonomy original
$ python3 run_goemotions.py --taxonomy group
$ python3 run_goemotions.py --taxonomy ekman
$ python3 run_goemotions.py --taxonomy all
```

If you want to use the CPCC regularizer, set "use_cpcc": true in `config/original.json`. Currently, we only support this regularizer at the level of original labels. It makes sense since at the higher level, the labels are much less correlated. 

To get visualizations, run
```
$ python3 visualize.py --taxonomy {$TAXONOMY}

$ python3 visualize.py --taxonomy original
$ python3 visualize.py --taxonomy group
$ python3 visualize.py --taxonomy ekman
$ python3 visualize.py --taxonomy all
```

