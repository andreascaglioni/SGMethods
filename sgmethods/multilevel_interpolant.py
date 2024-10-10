"""Module for a class that implements multilevel sparse grid interpolation.
Multilevel methods combine a parametric and physical approximation method in a
"smart" way in order to prodice a smaller error for the same number of degrees\
of freedom.

For a general introduction, see the classical paper on Mulilevel Monte Carlo:

*Giles MB. Multilevel Monte Carlo methods. Acta Numerica. 2015;24:259-328. 
doi:10.1017/S096249291500001X*, 
https://www.cambridge.org/core/journals/acta-numerica/article/abs/multilevel-monte-carlo-methods/C5AF9A57ED8FF8FDF08074C1071C5511
"""

class MLInterpolant:
    r"""Class for multilevel sparse grid interpolant as in 


    *Teckentrup, Jantsch, Webster, Gunzburger. A Multilevel Stochastic 
    Collocation Method for Partial Differential Equations with Random Input 
    Data, SIAM JUQ, 2015* https://epubs.siam.org/doi/abs/10.1137/140969002.

    We aim at approximating a function 
    :math:`u:\Gamma\times D\rightarrow \mathbb{R}`,
    where :math:`\Gamma\subset\mathbb{R}^N` is the parameter domain,
    :math:`D\subset\mathbb{R}^3` is the physical domain (a time variable can 
    also be included),
    and the codomain can be sobstituted by any Hilbert space.

    The multilevel interpolant with :math:`K+1` levels reads:

    .. math::

        u_{\text{ML}}^K = \sum_{k=0}^{K} (I_{K-k}-I_{K-k-1}) u_k,

    where :math:`I_{k}:\Gamma\rightarrow\mathbb{R}` is a sparse grid interpolat
    of "resolution" :math:`k=0,\dots, K`, and
    :math:`u_k:\Gamma\times D\rightarrow\mathbb{R}` is a space approximation 
    (e.g. finite elements)
    with "resolution" :math:`k=0,\dots, K`.
    """

    def __init__(self, interpolants_sequence):
        """Store the list of single-level interpolants (assumed to behave like
        :Class:`SGInterpolant`).

        Args:
            interpolants_sequence (list[:py:class:`sgmethods.sparse_grid_interpolant.SGInterpolant`]):
            List of sparse grid interpolants with appropriate multi-index sets.

        Returns:
            None
        """

        self.n_levels = len(interpolants_sequence)
        self.interp_seq = interpolants_sequence

    def sample(self, f_approx):
        """Compute multilevel samples of the function ``f`` on appopriate sparse
        grids noeed to compute the multilevel interpolant.

        Args:
            ``f_approx`` (Callable[[numpy.ndarray[float], int], float]):
                Function with 2 inputs: Parameters (2D numpy.ndarray[float], 
                where each row is a parameter vector), and the finite element
                accuracy indexed by :math:`k=0,\dots, K`.

        Returns:
            list[numpy.ndarray[float]]: List of the same length as
            ``self.interp_seq`` used to interpolate with :py:meth:`interpolate`.
            The k-th list element is a 2D numpy.ndarray[float] containing the 
            values of ``f_approx[k]`` (the level k finite element approximation)
            on the sparse grid of resolution K-k-1.
            Within this 2D array, each row corresponds to a point in the sparse
            grid at level K-k and consists of the coorodinates of the finite
            element approximation in that parameter.
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
            yy (numpy.ndarray[float]): 2D array. Each row is a parametric point.
            ml_samples_f (list[numpy.ndarray[float]]): Approrpiate list of 
                multilevel samples as computed by the method :py:meth:`sample`.

        Returns:
            numpy.ndarray[double]: 2D array. Each row is the approximation of
            ``f`` in the parameter given by the corresponding row of ``yy``.
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
        r"""Get the terms of the multi-level expansion split based on FE 
        approximation.

        Args:
            yy (numpy.array[float]): Each row is a parametric point where to 
                evaluate each of the multilevel terms.
            ml_samples_f (list[numpy.ndarray[float]]). List of 2D arrays.
                The k-th list entry contais data on 
                :math:`(I_{K-k}-I_{K-k-1})[u_k]`.
                Each row contains the finite element coordinates ina  point in 
                the level K-k sparse grid. 

        Returns:
            [list[numpy.ndarray[float]]]: List of 2D array. The k-th list entry
            contains the result for :math:`(I_{K-k}-I_{K-k-1})[u_k]`.
            Each row contains the finite element coordinates ina  point of `yy`.
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
        """Compute the cost of computing the multilevel interpolant.

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
