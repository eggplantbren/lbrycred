import numpy as np
import numpy.random as rng

NUM_CHANNELS = 4
TOL = 1E-6

def Phi(x):
    """
    Softening function
    """
    return (x + 1)**0.25

def PhiInv(x):
    """
    Inverse
    """
    return x**4 - 1.0


def update(y, w):
    """
    One iteration of the procedure to compute the ys.
    Returns True if converged.
    """
    ynew = y.copy()
    for j in range(NUM_CHANNELS):
        # These are the terms that make up eta_j in the document
        terms = ynew*w[:, j]
        terms[j] = w[j, j]
        ynew[j] = Phi(np.sum(terms))
    converged = np.mean(np.abs(ynew - y)) < TOL
    return ynew, converged

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
print("Calculating credibility scores...", flush=True, end="")
y = np.ones(NUM_CHANNELS)
iterations = 0
while True:
    y, converged = update(y, w)
    iterations += 1
    if converged:
        break
print(f"converged after {iterations} iterations.", end="\n\n")

print("Scores transformed on to LBC grade:")
print("-----------------------------------")
z = PhiInv(y)
z *= np.sum(w)/np.sum(z)
print(z)

