"""
Functions to implement belief propagation on a tree using the fast Python library Numba. 
The user should use the function run_BP to perform the inference (either of the root or of masked symbols)
"""

import numpy as np
from numba import njit
from multiprocessing import Pool,cpu_count

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
    
def run_BP(M,l,q,xis):
    # Convert the leaves into messages, not super efficient but sequences are not so long and just need to do it once
    leaves_BP = np.empty((len(xis),q))
    for i in range(len(xis)):
        if xis[i] == q + 1: # Masked symbols
            leaves_BP[i,:] = 1/q
        else:
            leaves_BP[i,:] = 0
            leaves_BP[i,xis[i]] = 1
    up_messages,down_messages = generate_tree(l,q,leaves_BP)
    up_messages,down_messages = update_messages(l,q,up_messages,down_messages,M)
    freeEntropy = get_freeEntropy(M,l,q,up_messages,down_messages)
    marginals = compute_marginals(l,q,up_messages,down_messages)
    return marginals,freeEntropy

def masked_inference(M,l,q,xis):
    marginals,_ = run_BP(M,l,q,xis)
    return marginals[-1,:,:]

def root_inference(M,l,q,xis):
    marginals,_ = run_BP(M,l,q,xis)
    return marginals[0,:,:]

def run_inference(M,l,q,xi,mask_rate):
    np.random.seed()
    xi_masked = np.copy(xi)
    masked_indices = np.random.choice(len(xi),size=int(mask_rate*len(xi)),replace=False)
    xi_masked[masked_indices] = q + 1
    marginals = masked_inference(M,l,q,xi_masked)
    success_rate = np.mean(np.argmax(marginals[masked_indices,:],axis=1) == xi[masked_indices])
    return success_rate

def MLM_BP_accuracy(M,l,q,xi,mask_rate,N_trials):
    p = Pool(cpu_count())
    runs = p.starmap(run_inference,[(M,l,q,xi[:,k],mask_rate) for k in range(int(N_trials))])
    success_rates = np.mean(runs)
    p.close()
    return success_rates