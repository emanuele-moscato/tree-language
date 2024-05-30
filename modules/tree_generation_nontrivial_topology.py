import numpy as np

class Grammar:
    def __init__(self, M, rho):
        self.M = M 
        self.rho = rho
    
def generate_grammar(Q=4, sigma=1., epsilon=0., rho_intermediate=0., max_depth=5):
    M = generate_M(Q=Q, sigma=sigma, epsilon=epsilon)
    rho = generate_rho(rho_intermediate=rho_intermediate, max_depth=max_depth) 
    return Grammar(M, rho)

def submat_softmax(C):
    Q = C.shape[0]
    M = np.zeros((Q,Q,Q))
    for q in range(Q):
        M[q] = np.exp(C[q]) / np.exp(C[q]).sum()
    return M
    
def generate_M(Q=4, sigma=1., epsilon=0.):
    C = np.random.randn(Q,Q,Q)
    partition_ind = np.arange(Q**2)
    np.random.shuffle(partition_ind)
    for root in range(Q):
        for ind in range(Q):
            pair = partition_ind[ind + root*Q]
            for r in range(Q):
                if r == root:
                    continue
                else: 
                    C[root, pair // Q, pair % Q] += np.log(epsilon)
    M = submat_softmax(C)
    return M
        

def generate_rho(rho_intermediate=0., max_depth=5): 
    rho = rho_intermediate * np.ones(max_depth+1)
    rho[0] = 0.; rho[-1] = 1.
    return rho

class Node:
    def __init__(self, value, parent=None, left=None, right=None):
        self.value = value 
        self.parent = parent
        self.left = left 
        self.right = right 

class Tree():
    def __init__(self, root):
        self.root = root
    
    def fill_leaves(self):
        leaves = []
        fill_leaves_recursive(self.root, leaves)
        self.leaves = leaves
        
def fill_leaves_recursive(n, leaves):
    if n.left is None:
        leaves.append(n)
    else: 
        fill_leaves_recursive(n.left, leaves)
        fill_leaves_recursive(n.right, leaves)
    
def generate_pattern(grammar):
    M, rho = grammar.M, grammar.rho
    Q = M.shape[0]
    
    root = Node(np.random.randint(Q))
    tree = Tree(root)
        
    depth = 0
    next_layer = [root]
    while len(next_layer) > 0:
        current_layer = next_layer
        next_layer = []
        for n in current_layer:
            if np.random.rand() < rho[depth]: # branch interrupts 
                continue
            else: # branch continues
                Mn = M[n.value]
                pair_ind = np.random.choice(Q**2, p=Mn.ravel())
                left_ind = pair_ind // Q 
                right_ind = pair_ind % Q 
                left = Node(left_ind, parent=n); n.left = left
                right = Node(right_ind, parent=n); n.right = right
                next_layer.append(left); next_layer.append(right)
        depth += 1
    
    tree.fill_leaves()
    return np.array([n.value for n in tree.leaves]), tree.root.value # (x, y) pair


def pad_sequence(sequence, max_seq_len, padding_token):
    """
    Pads the given sequence of symbols FROM THE RIGHT up to length
    `max_seq_len`.
    """
    return np.pad(
        sequence,
        (0, max_seq_len - len(sequence)),
        constant_values=padding_token
    )
