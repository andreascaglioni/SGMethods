python3 -m cProfile -o profile_myscript.prof -s 'cumtime' tests/test_sinW_adaptive.py > profile_myscript.txt
# to visualize .prof file, use snakeviz