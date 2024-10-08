# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path('..').resolve()))


from sgmethods.nodes_tp import tp_knots
from sgmethods.nodes_1d import equispaced_nodes
from scipy.special import erfinv


n_nodes = [3, 0, 1]
kk = tp_knots(equispaced_nodes, n_nodes)
print(kk)

print(erfinv(0.5))

exit()