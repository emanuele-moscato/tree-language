"""
Script to implement belief propagation on a tree using the fast Python library Numba. We will take advantage of the simple binary nature of the tree and of the problem to avoid using classes.

The script also serves as a data generator for the Root Inference with learning methods, as it also outputs the root and leaves symbols for each tree generated. It outputs: roots, leaves, free entropies from BP, the successes of the inference using BP (bools), the maximum marginal probability of the root symbol from BP, the M matrix used for the tree, and the parameters l, q, sigma, and epsilon. The data is saved in a .npy file for each set of parameters, containing the results from 1e4 trials for each of 32 different grammars.
"""

import numpy as np
from numba import njit
from multiprocessing import Pool,cpu_count
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax

def get_M(q,sigma,epsilon): # Refined prescription, no collisions but noise and log-normal distribution
    h = sigma*np.random.randn(q,q,q) + np.log(epsilon) # Actually parametrize with logits
    M = np.empty((q,q,q))
    tuples = np.array([(i,j) for i in range(q) for j in range(q)])
    np.random.shuffle(tuples)
    for i in range(q):
        for j in range(i*q,(i+1)*q):
            k,l = tuples[j]
            h[i,k,l] = sigma*np.random.randn()
        M[i,:,:] = softmax(h[i,:,:])
    return M

def get_leaves(x0,M,l):
    # Get the leaves of a tree of depth m
    def get_branches(S,M):
        # Get the two branches from a given state S
        p_flat = M[S,:,:].ravel()
        ind = np.arange(len(p_flat))
        return np.unravel_index(np.random.choice(ind, p=p_flat),M[S,:,:].shape)
    x = np.array([x0],dtype=np.int8) # Initialize the tree
    for i in range(1,l+1):
        x_new = np.empty(2**i,dtype=np.int8)
        for j in range(len(x)):
            x_new[2*j],x_new[2*j+1] = get_branches(x[j],M)
        x = x_new
    return x

@njit
def generate_tree(l,q,leaves):
    up_messages = np.zeros((l+1,2**l,q))
    down_messages = np.zeros((l+1,2**l,q))
    for i in range(l):
        for j in range(2**i):
            up_messages[i,j,:] = 1/q
            down_messages[i,j,:] = 1/q
    up_messages[l,:,:] = 1/q
    for j in range(2**l):
        down_messages[l,j,:] += leaves[j,:] # Add the prescribed leaves
    return up_messages, down_messages

@njit
def update_messages(l,q,up_messages,down_messages,M):
    # Pre allocate stuff
    r_up = np.zeros(q)
    l_up = np.zeros(q)
    v_down = np.zeros(q)
    # Start from the leaves and go up to update downgoing (root to leaves) messages
    for i in range(l-1,-1,-1):
        for j in range(2**i):
            l_down = down_messages[i+1,2*j,:]
            r_down = down_messages[i+1,2*j+1,:]
            # Update the outgoing messages
            v_down[:] = 0
            for p1 in range(q): # Not using @ because M matrix is not conitguous so better performance this way
                for p2 in range(q):
                    for p3 in range(q):
                        v_down[p1] += l_down[p2]*M[p1,p2,p3]*r_down[p3]
            down_messages[i,j,:] = v_down/np.sum(v_down)
    for i in range(l):
        for j in range(2**i):
            l_down = down_messages[i+1,2*j,:]
            r_down = down_messages[i+1,2*j+1,:]
            v_up = up_messages[i,j,:]
            # Update the outgoing messages
            r_up[:] = 0
            l_up[:] = 0
            v_down[:] = 0
            for p1 in range(q): # Not using @ because M matrix is not conitguous so better performance this way
                for p2 in range(q):
                    for p3 in range(q):
                        r_up[p1] += v_up[p2]*M[p2,p3,p1]*l_down[p3]
                        l_up[p1] += v_up[p2]*M[p2,p1,p3]*r_down[p3]
            up_messages[i+1,2*j,:] = l_up/np.sum(l_up)
            up_messages[i+1,2*j+1,:] = r_up/np.sum(r_up)
    return up_messages,down_messages

@njit
def compute_marginals(l,q,up_messages,down_messages):
    marginals = np.empty((l+1,2**l,q))
    for i in range(l+1):
        for j in range(2**i):
            marginals[i,j,:] = up_messages[i,j,:]*down_messages[i,j,:]
            marginals[i,j,:] = marginals[i,j,:]/np.sum(marginals[i,j,:])
    return marginals

@njit
def get_freeEntropy(M,l,q,up_messages,down_messages):
    # First compute the free entropy from the variables
    F_variables = 0
    for i in range(1,l): # Exclude both the root and the leaves
        for j in range(2**i):
            F_variables += np.log(np.sum(up_messages[i,j,:]*down_messages[i,j,:]))/np.log(q)
    # Now compute the free entropy from the factors
    F_factors = 0
    for i in range(l):
        for j in range(2**i):
            l_down = down_messages[i+1,2*j,:]
            r_down = down_messages[i+1,2*j+1,:]
            v_up = up_messages[i,j,:]
            z_factor = 0
            for p1 in range(q): # Not using @ because M matrix is not conitguous so better performance this way
                for p2 in range(q):
                    for p3 in range(q):
                        z_factor += v_up[p1]*M[p1,p2,p3]*l_down[p2]*r_down[p3]
            F_factors += np.log(z_factor)/np.log(q)
    return -(F_factors - F_variables)/2**l

def build_args(N_trials,M,l,q,encoder_1H):
    arg_tuples = [(M,l,q,encoder_1H) for i in range(N_trials)]
    return arg_tuples
    
def run_BP(M,l,q,encoder_1H):
    np.random.seed()
    x0 = np.random.randint(q)
    leaves = get_leaves(x0,M,l)
    leaves_1H = encoder_1H.transform(leaves.reshape(-1, 1))
    up_messages,down_messages = generate_tree(l,q,leaves_1H)
    up_messages,down_messages = update_messages(l,q,up_messages,down_messages,M)
    freeEntropy = get_freeEntropy(M,l,q,up_messages,down_messages)
    marginals = compute_marginals(l,q,up_messages,down_messages)
    if np.argmax(marginals[0,0,:]) == x0:
        success = True
    else:    
        success = False
    return x0,leaves,freeEntropy,success,np.max(marginals[0,0,:])

if __name__ == "__main__":
    p = Pool(cpu_count()//2) # Use half of the available cores
    N_trials = int(1e4)
    N_M = 32
    ls = [4,6,8,10]
    qs = [4,8,10]
    epsilons = np.logspace(-5,0,12)
    epsilons = np.insert(epsilons,0,0)
    sigmas = np.linspace(0,3,16)
    #encoder_1H = OneHotEncoder(sparse_output=False) # For recent versions of sklearn
    encoder_1H = OneHotEncoder(sparse=False) # For older version of sklearn
    for q in qs:
        encoder_1H.fit(np.arange(q).reshape(-1, 1))
        for l in ls:
            for k in range(len(sigmas)):
                for j in range(len(epsilons)):
                    free_entropies = np.empty((N_trials,N_M))
                    successes = np.empty((N_trials,N_M),dtype=bool)
                    max_marginals = np.empty((N_trials,N_M))
                    x0s = np.empty((N_trials,N_M),dtype=int)
                    xis = np.empty((int(2**l),N_trials,N_M),dtype=int)
                    M_s = np.empty((q,q,q,N_M))
                    for i in range(N_M):
                        np.random.seed(i)
                        M = get_M(q,sigmas[k],epsilons[j])
                        runs = p.starmap(run_BP,build_args(N_trials,M,l,q,encoder_1H))
                        for h in range(N_trials):
                            x0s[h,i],xis[:,h,i],free_entropies[h,i],successes[h,i],max_marginals[h,i] = runs[h]
                        M_s[:,:,:,i] = M
                    np.save('./sim_data/RootInference_{}_{}_{:.2f}_{:.5f}.npy'.format(q,l,sigmas[k],epsilons[j]),np.array([q,l,sigmas[k],epsilons[j],x0s,xis,free_entropies,successes,max_marginals,M_s],dtype=object))
                    print('q = {}, l = {}, sigma = {}, epsilon = {} done.'.format(q,l,sigmas[k],epsilons[j]))