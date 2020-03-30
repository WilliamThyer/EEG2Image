import matplotlib.pyplot as plt
import numpy as np
import math

def create_time_serie(size, time):
    """Generate a time serie of length size and dynamic with respect to time."""
    # Generating time-series
    support = np.arange(0, size)
    serie = np.sin(support + float(time))
    return serie

def tabulate(x, y, f):
    """Return a table of f(x, y). Useful for the Gram-like operations."""
    return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))

def cos_sum(a, b):
    """To work with tabulate."""
    return(math.cos(a+b))

def transform(serie):
    """Compute the Gramian Angular Field of an image"""
    # Min-Max scaling
    min_ = np.amin(serie)
    max_ = np.amax(serie)
    scaled_serie = (2*serie - max_ - min_)/(max_ - min_)

    # Floating point inaccuracy!
    scaled_serie = np.where(scaled_serie >= 1., 1., scaled_serie)
    scaled_serie = np.where(scaled_serie <= -1., -1., scaled_serie)

    # Polar encoding
    phi = np.arccos(scaled_serie)

    # GAF Computation (every term of the matrix)
    gaf = tabulate(phi, phi, cos_sum)

    return gaf, scaled_serie

def uni_to_gaf(serie=None,cmap=None):
    
    if serie is None:
        print("No time serie provided, using generated data!")
        serie = create_time_serie(45, 0)

    #transform and plot serie
    gaf, _ = transform(serie)
    if cmap is None:
        plt.matshow(gaf)
    else:
        plt.matshow(gaf,cmap=cmap)
    plt.axis('off')

