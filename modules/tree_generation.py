import numpy as np


def compute_rho_entropy(rho, q):
    entropy_normalization = 2. * np.log(q) / np.log(2.)

    return - np.sum([(rho[i, ...] * np.log(rho[i, ...])).sum() for i in range(3)]) / (3. * entropy_normalization)


def calcrho_lognormal(q, sigma):
    """
    Generates a tensor `rho` in which for each `a` (first index) each entry
    `rho[a, b, c]` (as the pair (b, c) varies) is sampled from a log-normal
    distribution. Again keeping `a` fixed, `rho[a, :, :]` is normalized to 1.
    This encodes the fact that `rho[a, b, c]` represents the conditional
    probability of generating an ordered pair of children nodes `(b, c)` from
    the parent `a`: `rho[a, b, c] = p((b, c) | a)`.
    """
    logrho=np.random.normal(0,size=(q,q,q))*sigma
    
    for a in np.arange(q):
        lrmax=np.max(logrho[a,:,:])
        logrho[a,:,:]=logrho[a,:,:]-lrmax
    rho=np.exp(logrho)

    # Normalize summing over the pairs (b, c) for each a (root).
    rho2=np.zeros((q,q))
    
    for a in np.arange(q):
        rho2=rho[a,:,:]
        sum2=np.sum(rho2)
        rho[a,:,:]=rho[a,:,:]/sum2
        
    return rho


def calcrho_simplified(q, eps=0.01):
    """
    """
    rho = eps * np.ones(shape=(q, q, q))

    for i in range(q):
        rho[i, i, i] = 1. - (q ** 2 -1) * eps

    return rho


def calcrho(matrix_type, **kwargs):
    """
    Handler method to generate different types of transition matrices.
    """
    if matrix_type == 'lognormal':
        rho = calcrho_lognormal(q=kwargs['q'], sigma=kwargs['sigma'])
    elif matrix_type == 'simplified':
        rho = calcrho_simplified(q=kwargs['q'], eps=kwargs['eps'])
    elif matrix_type == 'mixed':
        rho_s = calcrho('simplified', q=kwargs['q'], eps=kwargs['eps'])
        rho_l = calcrho('lognormal', q=kwargs['q'], sigma=kwargs['sigma'])

        # Fraction of "lognormality".
        mixing = kwargs['mixing']

        rho = ((1. - mixing) * rho_s + mixing * rho_l)
    else:
        raise NotImplementedError(
            f'Generation of transition matrices of type {matrix_type} not '
            'implemented'
        )
    
    return rho


def calcnn(b, c, q):
    """
    Converts the representation of an ordered pair of children nodes `(b, c)`
    from an object (b, c) in the Cartesian product {0, ..., q-1}x{0, ..., q-1}
    to an object in {0, ..., q ** 2 - 1}.
    """
    return b+q*c


def calcbc(nn, q):
    """
    Inverse map w.r.t. the `calcnn` function.
    """
    b=nn%q
    c=int((nn-b)/q)
    return b,c


def calcpsum(q,rho):
    """
    For each root node a, computes the cumulative sum of probabilities
    for children pairs (b, c).
    """
    p=np.zeros((q,q**2))
    psum=np.zeros((q,q**2))
    sumrule=np.zeros(q)

    # For each root a, computes the probabilities of children pair
    # (b, c) and stores them in p.
    for a in np.arange(q):
        for b in np.arange(q):
            for c in np.arange(q):
                nn=calcnn(b, c, q)

                # Probability of the ordered pair (b, c) given a.
                p[a,nn]=rho[a,b,c]
        sumrule=np.sum(p,axis=1)
    #print('sumrule',sumrule)

    # For each root a, computes the cumulative sum of the probabilities
    # of each children pair (b, c).
    for a in np.arange(q):
        psum[a,0]=p[a,0]
        for i in np.arange(q**2-1):
            j=i+1
            psum[a,j]=psum[a,j-1]+p[a,j]

    return psum


def genbc(q,a,psum):
    """
    Given a symbol a for a node, randomly generates an ordered pair
    of nodes (b, c) sampling from the distribution whose cumulative
    distribution sum is psum[a, :]. Essentially inverts the cumulative
    distribution function.
    """
    x=np.random.uniform(0,1)
    pp=np.zeros(q**2)
    pp=psum[a,:]
    for i in np.arange(q**2):
        #print('x,i,pp[i]',x,i,pp[i])
        if pp[i]>x:
            b,c=calcbc(i, q)
            return b,c
        

def gen_x(K,q,psum,root):
    """
    Generates a tree with K levels (after the root) given a vocabulary of
    size q, a transition tensor rho and a given root. Returns a matrix x of
    shape (K+1, N) (N = number of leaves = 2 ** K) where the first index
    refers to the level and the second to the nodes at that level. The matrix
    is rectangular despite the tree structure, therefore at level l, only the
    first 2 ** l values of the second index must be considered (the others are
    all zero by convention).
    """
    N=2**K  # Number of leaves.

    x=np.zeros((K+1,N)).astype(int)

    # Assign the root.
    x[0,0]=root

    # Generate each level of the tree.
    for ir in np.arange(K):
        # ir refers to the parents' level, r to the children's level.
        r=ir+1
        jnew=0
        for j in np.arange(2**ir):
            a=x[ir,j]  # Parent symbol.
            
            b,c=genbc(q,a,psum)  # Generated children symbols.
            #print('r=',r,'jnew,jnew+1=',jnew,jnew+1)
            x[r,jnew]=b
            x[r,jnew+1]=c

            # Increase by 2 because each parent generates a pair of children.
            jnew=jnew+2
    #print('itry=',itry,'sequence obtained at generation K=',K)
    #print(x[K,:])
    return x


def generate_dataset(rho, n_samples, k, q):
    """
    Generates a dataset of size `n_samples` given a tree structure with `k`
    level (exlcuding the root, so `k+1` levels in total), a vocabulary of size
    `q` and transition matrices `rho`. Roots are drawn from a uniform
    distribution.
    """
    # Generate the transition matrix.
    psum = calcpsum(q, rho)  # CDFs of the transition probabilities.

    # Generate roots.
    roots = np.random.choice(range(q), n_samples, replace=True)

    # Generate trees.
    trees = [
        gen_x(k, q, psum, root)
        for root in roots
    ]

    # Extract the leaves from the trees (for ease of use in the training
    # phase).
    leaves = np.array([
        tree[-1, :]
        for tree in trees
    ])

    return rho, trees, roots, leaves
