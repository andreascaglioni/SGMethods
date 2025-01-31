"""
The core module of SGmethods. It contains the implementation of sparse grid 
interpolation in a class.
"""

from multiprocessing import Pool
import numpy as np
from sgmethods.tp_interpolants import TPPwLinearInterpolator
from sgmethods.nodes_tp import tp_knots
from sgmethods.tp_interpolant_wrapper import TPInterpolatorWrapper

class SGInterpolant:
    """Sparse grid interpolant class. It stores all relevant information to 
    define it like multi-index set, 1D notes etc. 
    It automatically computes the sparse grid and inclusion-exclusion 
    coefficients upon initialization.
    It allows to interpolate high dimensinal functions on the sparse grid.
    """

    def __init__(self, mid_set, knots, lev2knots, \
                 tp_interpolant=TPPwLinearInterpolator, n_parallel=1, \
                    verbose=True):
        """Initialize data, compute inclusion-exclusion coeff.s, sparse grid

        Args:
            mid_set (numpy.ndarray[int]): 2D array. Multi-index set. NB must be
                downward closed.
            knots (Callable[[int], numpy.ndarray[double]]): Get the nodes vector
                with the given number of nodes.
            lev2knots (Callable[[int], int]): Given a level >=0, returns the
                corresponding number of nodes >0. 
            tp_interpolant (Class, optional): One of the classes in the module 
                tp_interpolants. (e.g. TPPwLinearInterpolator). The
                user can also define their own class following the instructions
                in the same module. Defaults to a piecewise-linear inteprolant.
            n_parallel (int, optional): Number of parallel computations. 
                Defaults to 1.
            verbose (bool, optional): Verbose output. Defaults to True.
        """
        self.verbose = verbose
        self.mid_set = mid_set  # np array of shape (#mids, N)
        self.card_mid_set = mid_set.shape[0]
        self.N = mid_set.shape[1]
        # NBB need knots[1] = 0 to increase number of dimensions
        self.knots = knots
        # NBB need lev2knots(0)=1 to increase number of dimensions
        self.lev2knots = lev2knots
        self.tp_interpolant = tp_interpolant

        self.combination_coeffs = []  # list of int
        self.active_mids = []  # list of np arrays
        self.active_tp_nodes_list = []  # list of tuples
        # which dimensions of currcent TP interpolant (in inclusion-exclusion
        # formula) are active (more than 1 node)
        self.active_tp_dims = []
        self.map_tp_to_SG = []  # list of np arrays of shape ()
        self.setup_interpolant()

        self.SG = []  # np array of shape (#colloc. pts, N)
        self.num_nodes = 0
        self.setup_SG()

        self.n_parallel = n_parallel

    def setup_interpolant(self):
        """ Computes and saves in class attributes some important quantities:
        combination coefficients, active multi-indicies, active TP dimensions,
        active TP nodes, map TP to SG nodes based on the multi-index set, the 
        nodes and the level-to-knot function.

        Args:
            None
        
        Returns:
            None
        """
        bookmarks = np.unique(self.mid_set[:, 0], return_index=True)[1]
        bk = np.hstack((bookmarks[2:], np.array(
            [self.card_mid_set, self.card_mid_set])))
        for n in range(self.card_mid_set):
            current_mid = self.mid_set[n, :]
            combin_coeff = 1
            # index of the 1st mid with 1st components = currentMid[0]+2 (dont
            # need to itertate over it or any following one)
            range_ids = bk[current_mid[0]]
            # NB in np slice start:stop, stop is NOT included!!
            mids_difference = self.mid_set[(n+1):(range_ids), :] - current_mid
            is_binary_vector = np.all(np.logical_and(
                mids_difference >= 0, mids_difference <= 1), axis=1)
            binary_rows = mids_difference[is_binary_vector]
            combi_elementary = np.power(-1, np.sum(binary_rows, axis=1))
            combin_coeff += np.sum(combi_elementary)

            if combin_coeff != 0:
                self.combination_coeffs.append(combin_coeff)
                self.active_mids.append(current_mid)
                num_nodes_dir = self.lev2knots(current_mid).astype(int)
                active_tp_dims = np.where(num_nodes_dir > 1)
                self.active_tp_dims.append(active_tp_dims[0])
                num_nodes_active_dirs = num_nodes_dir[active_tp_dims]
                # If no dimension is active, i.e. 1 cp, 1 need a length 1 array!
                if num_nodes_active_dirs.shape[0] == 0:
                    num_nodes_active_dirs = np.array([1])
                self.active_tp_nodes_list.append(
                    tp_knots(self.knots, num_nodes_active_dirs))
                shp = np.ndarray(tuple(num_nodes_active_dirs), dtype=int)
                self.map_tp_to_SG.append(shp)  # to be filled

    def setup_SG(self):
        """Computes and saves in class attributes some important quantities:
        SG (sparse grid), num_nodes, map_tp_to_SG
        based on active_tp_nodes_list.
        """

        SG = np.array([]).reshape((0, self.N))
        for n, curr_active_tp_nodes in enumerate(self.active_tp_nodes_list):
            curr_active_dims = self.active_tp_dims[n]
            # NB "*" is "unpacking" operator (return comma-separated list)
            mesh_grid = np.meshgrid(*curr_active_tp_nodes, indexing='ij')
            it = np.nditer(mesh_grid[0], flags=['multi_index'])
            for x in it:
                curr_node_active_dims = [mesh_grid[j][it.multi_index]
                                      for j in range(len(mesh_grid))]
                # complete it with 0s in inactive dimensions
                curr_node = np.zeros(self.N)
                curr_node[curr_active_dims] = curr_node_active_dims
                # check = np.where(~(SG-currNode).any(axis=1))[0]
                check = np.where(
                    np.sum(np.abs(SG-curr_node), axis=1) < 1.e-10)[0]
                found = check.shape[0]
                # if currNode was in the SG more than once, something wrong
                assert found <= 1
                if found:  # if found, add the index to mapTPtoSG[n]
                    self.map_tp_to_SG[n][it.multi_index] = check[0]
                else:  # if not found, add it to sparse grid, add to mapTPtoSG
                    SG = np.vstack((SG, np.array(curr_node)))
                    self.map_tp_to_SG[n][it.multi_index] = SG.shape[0]-1
        self.SG = SG
        self.num_nodes = SG.shape[0]

    def sample_on_SG(self, f, dim_f=None, old_xx=None, old_samples=None):
        """Sample a given function on the sparse grid. 
        
        Optionally recycle previous samples stored e.g. from a previous 
        interpolation. The class also takes care automatically of the case in
        which the sparse grid has increased dimension (number of approcimated 
        scaar parameters). 
        The class first checks whether or not there is anything to recycle, then
        it sample new values.

        NB The class assumes ``f`` takes as input a numpy.ndarray[float] of
        parameters. The 1st output is assumed to be a numpy.ndarray[float] or a
        float (in this case, it is transformed into a 1-entry 
        numpy.ndarray[float]).

        Args:
            f (Callable[[numpy.ndarray[float]], numpy.ndarray[float]]): The
                function to interpolate.
            dim_f (int, optional): Dimensinoality codomain of ``f``. Defaults to
                None.
            old_xx (numpy.ndarray[float], optional): 2D array.  Each row is a 
                parametric point Defaults to None.
            old_samples (numpy.ndarray[float], optional): 2D array. Each row
                corresponds to a row of old_xx. Defaults to None.

        Returns:
            numpy.ndarray[float]: 2D array. Each row
            is the value of ``f`` on a sparse grid (``self.SG``) point.
        """

        dim_f = -1  # TODO remove dimF from signature; check all other scripts

        # Find dimF (dimension of output of F). First try oldSamples; if empty,
        # sample on first SG node and add to oldSamples and oldXx
        assert  self.SG.shape[0] > 0
        if not old_samples is None:
            old_samples = np.atleast_1d(old_samples)
            dim_f = old_samples.shape[1]
            assert dim_f >= 1
        else:  # Sample on first SG node
            node0 = self.SG[0]
            f_on_SG0 = np.atleast_1d(f(node0))  # turn in array if scalar
            dim_f = f_on_SG0.size

        f_on_SG = np.zeros((self.num_nodes, dim_f))  # the return array
        # TODO change, if previou else needed, one sample is re-samples

        # Sanity checks dimensions oldXx, oldSamples
        if ((old_xx is None) or (old_samples is None)):
            old_xx = np.zeros((0, self.N))
            old_samples = np.zeros((0, dim_f))
        assert old_xx.shape[0] == old_samples.shape[0]
        assert old_samples.shape[1] == dim_f

        # Case 1: Cuurrent parameter space has larger dimension that oldXx.
        # Embed oldXx in space of self.SG by extending by 0
        if old_xx.shape[1] < self.SG.shape[1]:
            filler = np.zeros(\
                (old_xx.shape[0], self.SG.shape[1]-old_xx.shape[1])\
                    )
            old_xx = np.hstack((old_xx, filler))

        # Case 2: Current parameter space is smaller than that of oldXx.
        # Remove from oldXx parameters out of subspace, also adapt oldSamples
        elif old_xx.shape[1] > self.SG.shape[1]:
            curr_dim = self.N
            tail_norm = np.linalg.norm(
                old_xx[:, curr_dim::], ord=1, axis=1).astype(int)
            # last [0] so the result is np array
            valid_entries = np.where(tail_norm == 0)[0]
            old_xx = old_xx[valid_entries, 0:self.N]
            old_samples = old_samples[valid_entries]

        # Find parametric points to sample now
        n_recycle = 0
        to_compute = []
        yy_to_compute = []
        for n in range(self.num_nodes):
            curr_node = self.SG[n, :]
            check = np.where(np.linalg.norm(
                old_xx-curr_node, 1, axis=1) < 1.e-10)[0]
            if check.size > 1:
                print(check)
            assert check.size <= 1
            found = len(check)
            if found:
                f_on_SG[n, :] = old_samples[check[0], :]
                n_recycle += 1
            else:
                to_compute.append(n)
                yy_to_compute.append(curr_node)

        # Sample (possibily in parallel) on points found just above
        if len(to_compute) > 0:
            if self.n_parallel == 1:
                for i, idx in enumerate(to_compute):
                    f_on_SG[idx, :] = f(yy_to_compute[i])
            elif self.n_parallel > 1:
                pool = Pool(self.n_parallel)
                tmp = np.array(pool.map(f, yy_to_compute))
                pool.close()
                pool.join()
                if len(tmp.shape) == 1:
                    tmp = tmp.reshape((-1, 1))
                f_on_SG[to_compute, :] = tmp
            else:
                raise ValueError('self.NParallel not int >= 1"')
        if self.verbose:
            print("Recycled", n_recycle, \
                  "; Discarted", old_xx.shape[0]-n_recycle, \
                    "; Sampled", self.SG.shape[0]-n_recycle)
        return f_on_SG

    def interpolate(self, x_new, f_on_SG):
        """Evaluate the interpolant on new parametric points.

        Args:
            x_new (numpy.ndarray[float]): 2D array. New parametric points where
                to interpolate. Each row is a point.
            f_on_SG (numpy.ndarray[float]): 2D array. Values of function on the
                sparse grid. Each row is a value corresponding to a parametric
                point in the sparse grid ``self.SG``.

        Returns:
            numpy.ndarray[float]: 2D array. Values of the interpolant of f on
            ``x_new``. Each row is a value corresponds to the parameter stored
            in the same row of ``x_new``.
        """

        out = np.zeros((x_new.shape[0], f_on_SG.shape[1]))
        for n, mid in enumerate(self.active_mids):
            curr_active_nodes_tuple = self.active_tp_nodes_list[n]
            curr_active_dims = self.active_tp_dims[n]
            map_curr_tp_to_SG = self.map_tp_to_SG[n]
            # output is a matrix of shape = shape(mapCurrTPtoSG) + (dimF,)
            f_on_curr_tp_grid = f_on_SG[map_curr_tp_to_SG, :]
            tp_interpolant = TPInterpolatorWrapper(curr_active_nodes_tuple, \
                                      curr_active_dims, \
                                      f_on_curr_tp_grid, self.tp_interpolant)
            out = out + self.combination_coeffs[n] * tp_interpolant(x_new)
        return np.squeeze(out)  # remove dimensions of length 1
    
