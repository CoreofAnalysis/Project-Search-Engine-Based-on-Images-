
import numpy as np
import math
from decimal import Decimal

# --------->>>>
def cosine_similarity(vector_a, vector_b):
    """ Cosine similarity between two unit vectors. """
    return vector_a.dot(vector_b.T)

# --------->>>>
def chisq_similarity(vector_a, vector_b, eps=1e-10):
    """ Chi-Squared distance between two histograms (distributions). """
    return 0.5 * np.sum((vector_a - vector_b)**2 / (vector_a + vector_b + eps))

# --------->>>>
def intersection_similarity(vector_a, vector_b):
    """ Intersections of two histograms (distributions). """
    return np.sum(np.minimum(vector_a, vector_b)) / np.minimum(vector_a.sum(), vector_b.sum())

# --------->>>>
def cosine_similarity_1(vector_a, vector_b):
    """ Cosine similarity between two unit vectors. """
    dot = np.dot(vector_a, vector_b)
    norma = np.linalg.norm(vector_a)
    normb = np.linalg.norm(vector_b)
    return dot / (norma * normb)

# --------->>>>
def euclidean_similarity_1(vector_a, vector_b):
    """ Euclidean distance (similarity). """
    return np.linalg.norm(vector_a - vector_b)
 
# --------->>>>
def euclidean_similarity(x,y):
    """ Euclidean distance (similarity). """
    return math.sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))
 
# --------->>>>
def manhattan_similarity(x,y):
    """ Manhattan distance (similarity). """
    return sum(abs(a-b) for a,b in zip(x,y))

# --------->>>>
def nth_root(value, n_root):
    root_value = 1/float(n_root)
    return round (Decimal(value) ** Decimal(root_value),3)
 
def minkowski_distance(x,y,p_value):
    return nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),p_value)

# --------->>>>
def jaccard_similarity(x,y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)