import numpy as np
import numpy.random as rng

NUM_CHANNELS = 4

def Phi(x):
    """
    Softening function
    """
    return (x + 10.0)**0.25

def PhiInv(x):
    """
    Inverse
    """
    return x**4 - 10.0


def update(y, w):
    """
    One iteration of the procedure to compute the ys
    """
    for j in range(NUM_CHANNELS):
        # These are the terms that make up eta_j in the document
        terms = y*w[:, j]
        terms[j] = w[j, j]
        y[j] = Phi(np.sum(terms))
    return y

# Matrix of supports. Start with diagonal
w = np.zeros((NUM_CHANNELS, NUM_CHANNELS))
w += np.diag(np.exp(1.0 + rng.randn(NUM_CHANNELS)))

# Add a few signed supports (if they land off-diagonal that's what they are)
for r in range(1 + rng.randint(10)):
    i = rng.randint(NUM_CHANNELS)
    j = rng.randint(NUM_CHANNELS)
    w[i, j] += np.exp(3.0*rng.randn())

print("Support matrix:")
print("---------------")
print(w, end="\n\n")

print("Raw staked LBC:")
print("---------------")
print(np.sum(w, axis=0), end="\n\n")

# Credibilities
y = np.ones(NUM_CHANNELS)
#print("Credibility scores:")
#print("-------------------", end="\n\n")
#print("After 0 iterations:", y)
for i in range(10):
    y = update(y, w)
#    print(f"After {i+1} iterations:", y)
#print("\n\n")

print("Scores transformed on to LBC grade:")
print("-----------------------------------")
z = PhiInv(y)
z *= np.sum(w)/np.sum(z)
print(z)

