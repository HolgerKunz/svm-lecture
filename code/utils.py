import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

def scatter_2d(X, y, title):
    """Two-dimensional scatter plot

    Parameters
    ----------
    X : np.ndarray
        shape (n_samples, 2)
    y : np.ndarray
        labels (one-dimensional)
    title : str
        plot title

    Returns
    -------
    None
    """
    assert X.shape[1] == 2, 'X must be of shape (n_samples, 2)'
    plt.scatter(X[:, 0], X[:, 1],
                marker='o', s=50,
                c=y, edgecolors='None', alpha=0.35)
    plt.title(title)
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    plt.tick_params(axis='both',
                    top='off', bottom='off',
                    left='off', right='off')

def scatter_3d(X, y, title):
    """Three-dimensional scatter plot

    Parameters
    ----------
    X : np.ndarray
        shape (n_samples, 3)
    y : np.ndarray
        labels (one-dimensional)
    title : str
        plot title

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
    plt.title(title)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_zlabel('$x_2$')
    ax.view_init(elev=10)
    ax.set_axis_bgcolor('white')
