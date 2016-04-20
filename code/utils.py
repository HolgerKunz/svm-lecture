import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn


def to_r3(x0, x1):
    """Transform data in R**2 to R**3 using (x0, x1, x0**2 + x1**2)

    Parameters
    ----------
    x0 : np.ndarray
    x1 : np.ndarray

    Returns
    -------
    3-dimensional np.ndarray
    """
    assert isinstance(x0, np.ndarray) and isinstance(x1, np.ndarray)
    x2 = x0**2 + x1**2
    return np.column_stack((x0, x1, x2))

def scatter_2d(X, y):
    """Two-dimensional scatter plot

    Parameters
    ----------
    X : np.ndarray
        shape (n_samples, 2)
    y : np.ndarray
        labels (one-dimensional)

    Returns
    -------
    None
    """
    assert X.shape[1] == 2, 'X must be of shape (n_samples, 2)'
    plt.scatter(X[:, 0], X[:, 1],
                marker='o', s=50,
                c=y, edgecolors='None', alpha=0.35)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.tick_params(axis='both',
                    top='off', bottom='off',
                    left='off', right='off')

def scatter_3d(X, y):
    """Three-dimensional scatter plot

    Parameters
    ----------
    X : np.ndarray
        shape (n_samples, 3)
    y : np.ndarray
        labels (one-dimensional)

    Returns
    -------
    None
    """
    assert X.shape[1] == 3, 'X must be of shape (n_samples, 3)'
    fig = plt.figure(figsize=(20, 8)) 
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2],
               marker='o', s=25,
               c=y, edgecolors='None')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    ax.view_init(elev=10)
    ax.set_axis_bgcolor('white')

def maximal_margin_hyperplane(svc, X, y, margins=True, support_vectors=True):
    """Plot the maximal margin hyperplane

    Parameters
    ----------
    svc : sklearn.svm.classes.SVC
        a Scikit-Learn support vector classifier object
    X : np.ndarray
        shape (n_samples, 2)
    y : np.ndarray
        labels (one-dimensional)
    margins : bool
        controls whether or not to plot the margins
    support_vectors : bool
        controls whether or not the support vectors are identified

    Returns
    -------
    None

    Notes
    -----
    Code from http://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html
    """
    assert isinstance(svc, sklearn.svm.classes.SVC)
    assert X.shape[1] == 2, 'X must be of shape (n_samples, 2)'
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    w = svc.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(x_min * 2, x_max * 2)
    yy = a * xx - (svc.intercept_[0]) / w[1]
    b = svc.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = svc.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])
    plt.plot(xx, yy, color='DimGray', linestyle='-')
    if margins:
        plt.plot(xx, yy_down, color='DimGray', linestyle=':')
        plt.plot(xx, yy_up, color='DimGray', linestyle=':')
    if support_vectors:
        plt.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1],
                    s=100, facecolors='None', edgecolors='black')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

def soft_margin_hyperplane(svc, X, y):
    """Plot the soft margin hyperplane

    Notes
    -----
    Code from http://scikit-learn.org/stable/auto_examples/svm/plot_svm_margin.html
    """
    assert isinstance(svc, sklearn.svm.classes.SVC)
    assert X.shape[1] == 2, 'X must be of shape (n_samples, 2)'
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    w = svc.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(x_min * 2, x_max * 2)
    yy = a * xx - (svc.intercept_[0]) / w[1]
    margin = 1 / np.sqrt(np.sum(svc.coef_ ** 2))
    yy_down = yy + a * margin
    yy_up = yy - a * margin
    plt.plot(xx, yy, color='DimGray', linestyle='-')
    plt.plot(xx, yy_down, color='DimGray', linestyle=':')
    plt.plot(xx, yy_up, color='DimGray', linestyle=':')
