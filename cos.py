import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# import tensorflow as tf

# vectors
a = np.array([1,1])
b = np.array([1,-1])

# manually compute cosine similarity
dot = np.dot(a, b)
norma = np.linalg.norm(a)
normb = np.linalg.norm(b)
cos = dot / (norma * normb)

# use library, operates on sets of vectors
aa = a.reshape(1,2)
ba = b.reshape(1,2)
cos_lib =  cosine_similarity(aa, ba)

print(
    dot,
    norma,
    normb,
    cos,
    cos_lib
)
