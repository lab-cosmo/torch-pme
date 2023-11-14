"""
Created on Mon Jun  5 10:16:52 2023

@author: Kevin Huguenin-Dumittan
@author: Michele Ceriotti
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.special import spherical_jn


def _innerprod(xx, yy1, yy2):
    """
    Compute the inner product of two radially symmetric functions.

    Uses the inner product derived from the spherical integral without
    the factor of 4pi. Use simpson integration.

    Generates the integrand according to int_0^inf x^2*f1(x)*f2(x)
    """
    integrand = xx * xx * yy1 * yy2
    dx = xx[1] - xx[0]
    return (integrand[0] / 2 + integrand[-1] / 2 + np.sum(integrand[1:-1])) * dx


class RadialBasis:
    """
    Class for precomputing and storing all results related to the radial basis.

    These include:
    * A routine to evaluate the radial basis functions at the desired points
    * The transformation matrix between the orthogonalized and primitive
      radial basis (if applicable).

    All the needed splines that only depend on the hyperparameters
    are prepared as well by storing the values.

    Parameters
    ----------
    max_radial : int
        Number of radial functions
    max_angular : int
        Number of angular functions
    radial_basis_radius : float
        Environment cutoff
    radial_basis : str
        The radial basis. Currently implemented are
        'gto', 'gto_primitive', 'gto_normalized',
        'monomial_spherical', 'monomial_full'.
        For monomial: Only use one radial basis r^ell for each angular
        channel ell leading to a total of (lmax+1)^2 features.


    Attributes
    ----------
    radial_spline : scipy.interpolate.CubicSpline instance
        Spline function that takes in k-vectors (one or many) and returns
        the projections of the spherical Bessel function j_l(kr) onto the
        specified basis.
    center_contributions : array
        center_contributions
    orthonormalization_matrix : array
        orthonormalization_matrix
    """

    def __init__(
        self,
        max_radial,
        max_angular,
        radial_basis_radius,
        radial_basis,
        parameters=None,
    ):
        # Store the provided hyperparameters
        self.max_radial = max_radial
        self.max_angular = max_angular
        self.radial_basis_radius = radial_basis_radius
        self.radial_basis = radial_basis.lower()
        self.parameters = parameters

        # Orthonormalize
        self.compute_orthonormalization_matrix()

    def evaluate_primitive_basis_functions(self, xx):
        """
        Evaluate the basis functions prior to orthonormalization on a set
        of specified points xx.

        Parameters
        ----------
        xx : np.ndarray
            Radii on which to evaluate the (radial) basis functions

        Returns
        -------
        yy : np.ndarray
            Radial basis functions evaluated on the provided points xx.

        """
        # Define shortcuts
        nmax = self.max_radial
        lmax = self.max_angular
        rcut = self.radial_basis_radius

        # Initialization
        yy = np.zeros((lmax + 1, nmax, len(xx)))

        # Initialization
        if self.radial_basis in ["gto", "gto_primitive", "gto_normalized"]:
            # Generate length scales sigma_n for R_n(x)
            sigma = np.ones(nmax, dtype=float)
            for i in range(1, nmax):
                sigma[i] = np.sqrt(i)
            sigma *= rcut / nmax

            # Define primitive GTO-like radial basis functions
            def f_gto(n, x):
                return x**n * np.exp(-0.5 * (x / sigma[n]) ** 2)

            R_n = np.array([f_gto(n, xx) for n in range(nmax)])  # nmax x Nradial

            # In this case, all angular channels use the same radial basis
            for ell in range(lmax + 1):
                yy[ell] = R_n

        elif self.radial_basis == "monomial_full":
            for ell in range(lmax + 1):
                for n in range(nmax):
                    yy[ell, n] = xx**n

        elif self.radial_basis == "monomial_spherical":
            for ell in range(lmax + 1):
                for n in range(nmax):
                    yy[ell, n] = xx ** (ell + 2 * n)

        elif self.radial_basis == "spherical_bessel":
            for ell in range(lmax + 1):
                # Define target function and the estimated location of the
                # roots obtained from the asymptotic expansion of the
                # spherical Bessel functions for large arguments x
                def f(x, ell):
                    return spherical_jn(ell, x)

                roots_guesses = np.pi * (np.arange(1, nmax + 1) + ell / 2)

                # Compute roots from initial guess using Newton method
                for n, root_guess in enumerate(roots_guesses):
                    root = fsolve(f, root_guess, args=(ell,))[0]
                    yy[ell, n] = spherical_jn(ell, xx * root / rcut)

        else:
            raise ValueError("Radial basis is not supported!")

        return yy

    def compute_orthonormalization_matrix(self, Nradial=5000):
        """
        Compute orthonormalization matrix for the specified radial basis

        Parameters
        ----------
        Nradial : int, optional
            Number of nodes to be used in the numerical integration.

        Returns
        -------
        None.
        It stores the precomputed orthonormalization matrix as part of the
        class for later use, namely when calling
        "evaluate_radial_basis_functions"

        """
        # Define shortcuts
        nmax = self.max_radial
        lmax = self.max_angular
        rcut = self.radial_basis_radius

        # Evaluate radial basis functions
        xx = np.linspace(0, rcut, Nradial)
        yy = self.evaluate_primitive_basis_functions(xx)

        # Gram matrix (also called overlap matrix or inner product matrix)
        innerprods = np.zeros((lmax + 1, nmax, nmax))
        for ell in range(lmax + 1):
            for n1 in range(nmax):
                for n2 in range(nmax):
                    innerprods[ell, n1, n2] = _innerprod(xx, yy[ell, n1], yy[ell, n2])

        # Get the normalization constants from the diagonal entries
        self.normalizations = np.zeros((lmax + 1, nmax))
        for ell in range(lmax + 1):
            for n in range(nmax):
                self.normalizations[ell, n] = 1 / np.sqrt(innerprods[ell, n, n])
                innerprods[ell, n, :] *= self.normalizations[ell, n]
                innerprods[ell, :, n] *= self.normalizations[ell, n]

        # Compute orthonormalization matrix
        self.transformations = np.zeros((lmax + 1, nmax, nmax))
        for ell in range(lmax + 1):
            eigvals, eigvecs = np.linalg.eigh(innerprods[ell])
            self.transformations[ell] = (
                eigvecs @ np.diag(np.sqrt(1.0 / eigvals)) @ eigvecs.T
            )

    def evaluate_radial_basis_functions(self, nodes):
        """
        Evaluate the orthonormalized basis functions at specified nodes.

        Parameters
        ----------
        nodes : np.ndarray of shape (N,)
            Points (radii) at which to evaluate the basis functions.

        Returns
        -------
        yy_orthonormal : np.ndarray of shape (lmax+1, nmax, N,)
            Values of the orthonormalized radial basis functions at each
            of the provided points (nodes).

        """
        # Define shortcuts
        lmax = self.max_angular
        nmax = self.max_radial

        # Evaluate the primitive basis functions
        yy_primitive = self.evaluate_primitive_basis_functions(nodes)

        # Convert to normalized form
        yy_normalized = yy_primitive
        for ell in range(lmax + 1):
            for n in range(nmax):
                yy_normalized[ell, n] *= self.normalizations[ell, n]

        # Convert to orthonormalized form
        yy_orthonormal = np.zeros_like(yy_primitive)
        for ell in range(lmax + 1):
            for _ in range(nmax):
                yy_orthonormal[ell, :] = (
                    self.transformations[ell] @ yy_normalized[ell, :]
                )

        return yy_orthonormal
