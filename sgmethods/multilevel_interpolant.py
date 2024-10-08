"""Module for a class that implements the multilevel sparse grid interpolation
from  `[Teckentrup, Jantsch, Webster, Gunzburger (2015)] 
<https://epubs.siam.org/doi/abs/10.1137/140969002>`_.
"""

class MLInterpolant:
    """Class for multilevel sparse grid interpolant.
    """

    def __init__(self, interpolants_sequence):
        """Store the list of single-level interpolants (assumed to behave like
        :Class:`SGInterpolant`).

        Args:
            interpolants_sequence (list): List of sparse grid interpolants with
            appropriate multi-index sets.

        Returns:
            None
        """

        self.n_levels = len(interpolants_sequence)
        self.interp_seq = interpolants_sequence

    def sample(self, f_approx):
        """Sample ``f_approx`` on the collocation nodes needed to compute the
        multileve interpolant.

        Args:
            ``f_approx`` (Callable[[numpy.ndarray[float], int], float]):
                Function with 2 inputs: A parameter (1D double array of any 
                length), and a level ``k`` (int >=0) that gives approximation 
                accuracy level.

        Returns:
            list: List of the same length as ``interpolants_sequence`` used to
            construct the object. Each list element is a 2D numpy.ndarray. Each
            row is a sample from ``f_approx`` for a parameter in the 
            corresponding sparse grid.
        """

        # TODO: FApprox is only allowed to return values that are 1D arrays of
        #       FIXED length, indepednet of the level
        # TODO: Assert that FAppeox return onyly arrays of fixed length;
        ml_samples_f = []
        for k in range(self.n_levels):
            def f_approx_curr_level(y):
                return f_approx(y, k)
            samples_current_level = self.interp_seq[self.n_levels-1-k].\
                sampleOnSG(f_approx_curr_level)
            ml_samples_f.append(samples_current_level)
        return ml_samples_f

    def interpolate(self, yy, ml_samples_f):
        """Use the interpolation operator to approximate the function ``f`` of
        which ML samples are given.

        Args:
            yy (numpy.ndarray[float]): Each row is a parameter of any length. 
                The number of rows is also arbitrary.
            ml_samples_f (list): List of multilevel samples as computed by the
                method :py:meth:`sample`.

        Returns:
            numpy.ndarray[double]: Each row is the approximation of ``f`` in the 
            parameter given by the correspondg row of `yy`.
        """

        interpolant_on_yy = self.interp_seq[0].interpolate(
            yy, ml_samples_f[self.n_levels-1])  # first hande out of loop
        for k in range(1, self.n_levels):
            samples_curr = ml_samples_f[self.n_levels-1-k]
            def f_mock(y):
                return 1/0  # TODO write something more decent
            samples_curr_reduced = self.interp_seq[k-1].sampleOnSG(
                f_mock, oldXx=self.interp_seq[k].SG, oldSamples=samples_curr)
            interpolant_on_yy += \
                self.interp_seq[k].interpolate(yy, samples_curr) - \
                self.interp_seq[k-1].interpolate(yy, samples_curr_reduced)
        return interpolant_on_yy

    def get_ml_terms(self, yy, ml_samples_f):
        r"""Get terms of multi-level expansion split based on FE spaces.

        Args:
            yy (numpy.array[float]): each row is a parameter vector toe valuate
                MLSamplesF (list of 2D arrays double). The k-th entry is a 2D 
                array with shape `nY` x kth physical space size representing 
                :math:`(I_{K-k}-I_{K-k-1})[u_k]`.

        Returns:
            [list]: k-th term is 2D array. Each row corresponds to a parameter 
            in `yy`. Each colum gives the finite element corrdinates.
        """

        ml_terms = []
        for k in range(self.n_levels):
            curr_sg_level = self.n_levels-1-k
            curr_vals = \
                self.interp_seq[curr_sg_level].interpolate(yy, ml_samples_f[k])
            # NBB the last FE level coresponds to 1 interpolant since I_{-1}=0
            if k < self.n_levels-1:
                def f_mock(x):
                    return 0.
                ml_samples_reduceds = self.interp_seq[curr_sg_level-1].\
                    sampleOnSG(f_mock, oldXx=self.interp_seq[curr_sg_level].SG,
                               oldSamples=ml_samples_f[k])
                curr_vals -= self.interp_seq[curr_sg_level -
                                      1].interpolate(yy, ml_samples_reduceds)
            ml_terms.append(curr_vals)
        return ml_terms

    def total_cost(self, cost_kk):
        """Compute cost of computing this ML appeoximation.

        Args:
            cost_kk (numpy.ndarray[float]): The k-th entry is the cost of 
                computing 1 finite element sample at level k.

        Returns:
            float: Total cost based on number of sparse grid nodes, levels, and
            finite element computations.
        """

        assert cost_kk.size == self.n_levels
        total_cost_curr = 0
        for k in range(self.n_levels):
            total_cost_curr += \
                self.interp_seq[k].numNodes * (cost_kk[self.n_levels-k-1])
        return total_cost_curr
