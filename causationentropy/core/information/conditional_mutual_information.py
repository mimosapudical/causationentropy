import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import digamma

from causationentropy.core.information.entropy import (
    geometric_knn_entropy,
    kde_entropy,
    poisson_entropy,
    poisson_joint_entropy,
)
from causationentropy.core.information.mutual_information import (
    gaussian_mutual_information,
    geometric_knn_mutual_information,
    kde_mutual_information,
    knn_mutual_information,
)


def gaussian_conditional_mutual_information(X, Y, Z=None):
    r"""
    Compute conditional mutual information for multivariate Gaussian variables.

    For multivariate Gaussian variables, the conditional mutual information has
    a closed-form expression using covariance matrix determinants:

    .. math::

        I(X; Y | Z) = \frac{1}{2} \log \frac{|\Sigma_{XZ}| |\Sigma_{YZ}|}{|\Sigma_Z| |\Sigma_{XYZ}|}

    This can also be expressed as:

    .. math::

        I(X; Y | Z) = \frac{1}{2} [\log |\Sigma_{XZ}| + \log |\Sigma_{YZ}| - \log |\Sigma_Z| - \log |\Sigma_{XYZ}|]

    where :math:`\Sigma_{\cdot}` denotes the covariance matrix of the subscripted variables.

    Parameters
    ----------
    X : array-like of shape (N, k_x)
        First variable with N samples and k_x features.
    Y : array-like of shape (N, k_y)
        Second variable with N samples and k_y features.
    Z : array-like of shape (N, k_z) or None
        Conditioning variable with N samples and k_z features.
        If None, computes marginal mutual information I(X;Y).

    Returns
    -------
    I : float
        Conditional mutual information in nats.

    Notes
    -----
    This implementation uses log-determinants of correlation matrices for
    numerical stability, employing the signed log-determinant function
    to handle potential numerical issues.

    The Gaussian assumption implies that:
    - All conditional dependencies are captured by linear relationships
    - Higher-order moments beyond covariance carry no information
    - The estimator is exact under Gaussianity

    For non-Gaussian data, this estimator provides a lower bound on the
    true conditional mutual information.
    """
    if Z is None:
        return gaussian_mutual_information(X, Y)

    def _detcorr(A):
        C = np.corrcoef(A.T)
        # For 1D input, corrcoef returns scalar 1.0, and log(1.0) = 0.0
        return 0.0 if np.ndim(C) == 0 else np.linalg.slogdet(C)[1]

    SZ = _detcorr(Z)
    SXZ = _detcorr(np.hstack((X, Z)))
    SYZ = _detcorr(np.hstack((Y, Z)))
    SXYZ = _detcorr(np.hstack((X, Y, Z)))

    cmi = 0.5 * (SXZ + SYZ - SZ - SXYZ)
    return cmi


def kde_conditional_mutual_information(
    X, Y, Z, bandwidth="silverman", kernel="gaussian"
):
    """
    Estimate conditional mutual information using Kernel Density Estimation.

    This function computes conditional mutual information using the entropy decomposition:

    .. math::

        I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)

    where each entropy term is estimated using kernel density estimation.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features_x)
        First variable.
    Y : array-like of shape (n_samples, n_features_y)
        Second variable.
    Z : array-like of shape (n_samples, n_features_z) or None
        Conditioning variable. If None, reduces to marginal mutual information.
    bandwidth : str or float, default='silverman'
        Bandwidth parameter for KDE.
    kernel : str, default='gaussian'
        Kernel function for density estimation.

    Returns
    -------
    I : float
        Estimated conditional mutual information in nats.

    Notes
    -----
    The KDE approach can capture nonlinear conditional dependencies but suffers from:
    - Curse of dimensionality for high-dimensional conditioning sets
    - Bandwidth selection sensitivity
    - Computational complexity scaling with sample size

    Consider k-NN methods for high-dimensional problems or large datasets.
    """
    if Z is None:
        I = kde_mutual_information(X, Y, bandwidth=bandwidth, kernel=kernel)
    else:
        XZ = np.hstack((X, Z))
        YZ = np.hstack((Y, Z))
        XYZ = np.hstack((X, Y, Z))

        # Compute the entropies
        Hz = kde_entropy(Z, bandwidth=bandwidth, kernel=kernel)
        Hxz = kde_entropy(XZ, bandwidth=bandwidth, kernel=kernel)
        Hyz = kde_entropy(YZ, bandwidth=bandwidth, kernel=kernel)
        Hxyz = kde_entropy(XYZ, bandwidth=bandwidth, kernel=kernel)
        I = Hxz + Hyz - Hxyz - Hz

    return I


def knn_conditional_mutual_information(X, Y, Z, metric="minkowski", k=1):
    """
    Estimate conditional mutual information using k-nearest neighbor method.

    This function implements conditional mutual information estimation using
    the relationship:

    .. math::

        I(X; Y | Z) = I(X, Y) - I(X, Y; Z)

    where both mutual information terms are estimated using the KSG k-NN estimator.

    The approach leverages the fact that:

    .. math::

        I(X; Y | Z) = I(X; Y) - I(X; Y | Z)

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features_x)
        First variable.
    Y : array-like of shape (n_samples, n_features_y)
        Second variable.
    Z : array-like of shape (n_samples, n_features_z) or None
        Conditioning variable. If None, computes marginal mutual information.
    metric : str, default='minkowski'
        Distance metric for k-NN calculations.
    k : int, default=1
        Number of nearest neighbors.

    Returns
    -------
    I : float
        Estimated conditional mutual information in nats.

    Notes
    -----
    This implementation uses the decomposition approach rather than direct
    conditional MI estimation. The accuracy depends on:

    - Quality of marginal MI estimates
    - Dimensionality of the joint space
    - Sample size relative to effective dimensionality

    References
    ----------
    .. [1] Kraskov, A., Stögbauer, H., Grassberger, P. Estimating mutual information.
           Physical Review E 69, 066138 (2004).
    """
    if Z is None:
        return knn_mutual_information(X, Y, metric=metric, k=k)
    else:
        JS = np.column_stack((X, Y, Z))
        # Find the K-th smallest distance in the joint space
        if metric == "minkowski":
            D = np.sort(cdist(JS, JS, metric=metric, p=k + 1), axis=1)[:, k]
        else:
            D = np.sort(cdist(JS, JS, metric=metric), axis=1)[:, k]
        epsilon = D
        # Count neighbors within epsilon in marginal spaces
        Dxz = cdist(np.column_stack((X, Z)), np.column_stack((X, Z)), metric=metric)
        nxz = np.sum(Dxz < epsilon[:, None], axis=1) - 1
        Dyz = cdist(np.column_stack((Y, Z)), np.column_stack((Y, Z)), metric=metric)
        nyz = np.sum(Dyz < epsilon[:, None], axis=1) - 1
        Dz = cdist(Z, Z, metric=metric)
        nz = np.sum(Dz < epsilon[:, None], axis=1) - 1

        # VP Estimation formula
        I = digamma(k) - np.mean(digamma(nxz + 1) + digamma(nyz + 1) - digamma(nz + 1))
        return I


def geometric_knn_conditional_mutual_information(X, Y, Z, metric="euclidean", k=1):
    """
    Estimate conditional mutual information using geometric k-nearest neighbor method.

    This function applies the geometric k-NN entropy estimator to compute
    conditional mutual information via the entropy decomposition:

    .. math::

        I(X; Y | Z) = H_{\text{geom}}(X, Z) + H_{\text{geom}}(Y, Z) - H_{\text{geom}}(Z) - H_{\text{geom}}(X, Y, Z)

    The geometric correction accounts for local manifold structure, providing
    improved estimates for data with non-uniform density or intrinsic dimensionality
    lower than the ambient space.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features_x)
        First variable.
    Y : array-like of shape (n_samples, n_features_y)
        Second variable.
    Z : array-like of shape (n_samples, n_features_z) or None
        Conditioning variable. If None, computes marginal mutual information.
    metric : str, default='euclidean'
        Distance metric for neighbor calculations.
    k : int, default=1
        Number of nearest neighbors.

    Returns
    -------
    I : float
        Estimated conditional mutual information using geometric k-NN method.

    Notes
    -----
    The geometric approach is particularly effective for:
    - Data on lower-dimensional manifolds
    - Non-uniform density distributions
    - Cases where local geometric structure is important

    The method accounts for the effective local dimensionality through
    geometric corrections to the standard k-NN entropy estimates.

    References
    ----------
    .. [1] Lord, W.M., Sun, J., Bollt, E.M. Geometric k-nearest neighbor estimation of
           entropy and mutual information. Chaos 28, 033113 (2018).
    """

    if Z is None:
        return geometric_knn_mutual_information(X, Y)
    YZdist = cdist(np.hstack((Y, Z)), np.hstack((Y, Z)), metric=metric)
    XZdist = cdist(np.hstack((X, Z)), np.hstack((X, Z)), metric=metric)
    XYZdist = cdist(np.hstack((X, Y, Z)), np.hstack((X, Y, Z)), metric=metric)
    Zdist = cdist(Z, Z, metric=metric)
    HZ = geometric_knn_entropy(Z, Zdist, k)
    HXZ = geometric_knn_entropy(np.hstack((X, Z)), XZdist, k)
    HYZ = geometric_knn_entropy(np.hstack((Y, Z)), YZdist, k)
    HXYZ = geometric_knn_entropy(np.hstack((X, Y, Z)), XYZdist, k)
    cmi = HXZ + HYZ - HXYZ - HZ
    return cmi


def _poisson_rate_matrix_from_correlation(correlation):
    """Convert the paper's correlation surrogate into Poisson component rates.

    Fish, Sun, and Bollt estimate the off-diagonal shared rates from pairwise
    correlations so that the approximate rates remain in the small-rate regime.
    Their Eq. (46) then recovers each private rate by subtracting the sum of the
    shared rates from the corresponding diagonal entry.
    """
    correlation = np.atleast_2d(
        np.asarray(correlation, dtype=float)
    )
    shared = correlation - np.diag(np.diag(correlation))
    rates = shared.copy()
    private = np.diag(correlation) - np.sum(shared, axis=1)
    np.fill_diagonal(rates, private)
    return rates


def _poisson_rate_matrix(samples):
    """Estimate the scaled Poisson component-rate matrix from observations."""
    samples = np.asarray(samples)
    correlation = np.corrcoef(samples, rowvar=False)
    return _poisson_rate_matrix_from_correlation(correlation)


def _poisson_joint_entropy_from_samples(samples):
    """Estimate joint Poisson entropy using the paper's correlation scaling."""
    return poisson_joint_entropy(_poisson_rate_matrix(samples))


def poisson_conditional_mutual_information(X, Y, Z):
    """
    Estimate conditional mutual information for multivariate Poisson variables.

    The estimator follows Fish, Sun, and Bollt (2022). Their network experiments
    intentionally use pairwise correlations as scaled surrogates for the shared
    Poisson component rates, then recover the private rates with Eq. (46). For
    conditional mutual information, Eq. (38) requires the proper Poisson
    marginals in each entropy term:

        I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z).

    Re-estimating the rate matrix for each retained variable set automatically
    absorbs shared components involving marginalized variables into the retained
    variables' private rates, as required by the Poisson marginal construction.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features_x)
        Count data from the first Poisson variables.
    Y : array-like of shape (n_samples, n_features_y)
        Count data from the second Poisson variables.
    Z : array-like of shape (n_samples, n_features_z) or None
        Conditioning variables. If None, computes marginal mutual information.

    Returns
    -------
    I : float
        Estimated conditional mutual information for Poisson data.

    References
    ----------
    Fish, J., Sun, J. & Bollt, E. Interaction networks from discrete event
    data by Poisson multivariate mutual information estimation and information
    flow with applications from gene expression data. Applied Network Science
    7, 70 (2022). https://doi.org/10.1007/s41109-022-00510-x
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    if Z is None:
        H_X = _poisson_joint_entropy_from_samples(X)
        H_Y = _poisson_joint_entropy_from_samples(Y)
        H_XY = _poisson_joint_entropy_from_samples(np.hstack((X, Y)))
        return H_X + H_Y - H_XY

    Z = np.atleast_2d(Z)
    H_XZ = _poisson_joint_entropy_from_samples(np.hstack((X, Z)))
    H_YZ = _poisson_joint_entropy_from_samples(np.hstack((Y, Z)))
    H_XYZ = _poisson_joint_entropy_from_samples(np.hstack((X, Y, Z)))
    H_Z = _poisson_joint_entropy_from_samples(Z)
    return H_XZ + H_YZ - H_XYZ - H_Z


def conditional_mutual_information(
    X,
    Y,
    Z=None,
    method="gaussian",
    metric="euclidean",
    k=6,
    bandwidth="silverman",
    kernel="gaussian",
):
    """
    Compute conditional mutual information using specified estimation method.

    This function provides a unified interface for computing conditional mutual information
    I(X;Y|Z) using various estimation approaches. The choice of method depends on the
    data type, dimensionality, and distributional assumptions.

    Conditional mutual information quantifies the information shared between X and Y
    when conditioning on Z:

    .. math::

        I(X; Y | Z) = H(X | Z) - H(X | Y, Z)

    Equivalently:

    .. math::

        I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features_x)
        First variable.
    Y : array-like of shape (n_samples, n_features_y)
        Second variable.
    Z : array-like of shape (n_samples, n_features_z) or None
        Conditioning variable. If None, computes marginal mutual information I(X;Y).
    method : str, default='gaussian'
        Estimation method. Available options:

        - 'gaussian': Assumes multivariate Gaussian distributions
        - 'kde' or 'kernel_density': Kernel density estimation
        - 'knn': k-nearest neighbor (KSG) estimator
        - 'geometric_knn': Geometric k-NN with manifold corrections
        - 'poisson': For discrete count data with Poisson assumptions

    metric : str, default='euclidean'
        Distance metric for k-NN based methods.
    k : int, default=1
        Number of nearest neighbors for k-NN methods.
    bandwidth : str or float, default='silverman'
        Bandwidth parameter for KDE methods.
    kernel : str, default='gaussian'
        Kernel function for KDE methods.

    Returns
    -------
    I : float
        Estimated conditional mutual information in nats.

    Raises
    ------
    ValueError
        If an unsupported method is specified.

    Notes
    -----
    **Method Selection Guidelines:**

    - **Gaussian**: Best for linear relationships, exact under Gaussianity
    - **KDE**: Good for smooth nonlinear dependencies, curse of dimensionality
    - **k-NN**: Robust for moderate dimensions, adapts to local density
    - **Geometric k-NN**: Effective for manifold data with intrinsic structure
    - **Poisson**: Specifically for discrete count data

    **Computational Complexity:**
    - Gaussian: O(n³) for matrix operations
    - KDE: O(n²) for density evaluation
    - k-NN: O(n² log n) for neighbor finding

    **Sample Size Requirements:**
    - Increase with dimensionality and complexity of dependencies
    - k-NN methods generally require fewer samples than KDE
    - Parametric methods (Gaussian) most sample-efficient when assumptions hold

    Examples
    --------
    >>> import numpy as np
    >>> from causationentropy.core.information.conditional_mutual_information import conditional_mutual_information
    >>>
    >>> # Generate sample data
    >>> n = 1000
    >>> X = np.random.randn(n, 2)
    >>> Y = np.random.randn(n, 1)
    >>> Z = np.random.randn(n, 1)
    >>>
    >>> # Compute conditional MI using different methods
    >>> cmi_gauss = conditional_mutual_information(X, Y, Z, method='gaussian')
    >>> cmi_knn = conditional_mutual_information(X, Y, Z, method='knn', k=3)
    >>>
    >>> print(f"Gaussian CMI: {cmi_gauss:.3f}")
    >>> print(f"k-NN CMI: {cmi_knn:.3f}")
    """
    if method == "gaussian":
        cmi = gaussian_conditional_mutual_information(X, Y, Z)

    elif method == "kde" or method == "kernel_density":
        cmi = kde_conditional_mutual_information(
            X, Y, Z, bandwidth=bandwidth, kernel=kernel
        )

    elif method == "knn":
        cmi = knn_conditional_mutual_information(X, Y, Z, metric=metric, k=k)

    elif method == "geometric_knn":
        cmi = geometric_knn_conditional_mutual_information(X, Y, Z, metric=metric, k=k)

    elif method == "poisson":
        cmi = poisson_conditional_mutual_information(X, Y, Z)

    else:
        supported_methods = [
            "gaussian",
            "kde",
            "kernel_density",
            "knn",
            "geometric_knn",
            "poisson",
        ]
        raise ValueError(
            f"Method '{method}' unavailable. Supported methods: {supported_methods}"
        )

    # Ensure non-negativity: CMI is theoretically always >= 0,
    # but finite sample estimation can produce small negative values.
    # Only clamp finite values; preserve NaN/inf for error handling.
    if np.isfinite(cmi):
        return max(0.0, cmi)
    return cmi
