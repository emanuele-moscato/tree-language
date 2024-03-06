using Einsum
using StatsBase
using Distributions
using ExtractMacro
using Statistics

xlogx(x) = x == 0. ? 0. : x * log(x) 
xlog2x(x) = x == 0. ? 0. : x * log2(x) 

# implemented:
#- lognormal(σ)
#- lognormal with parameters {σ, η1, η2} controlling:
#   - η1 ∈ [0,1] = symmetry / asymmetry
#   - η2 ∈ [0,1] = root dependence
#   - σ : width of the normal distribution
function generate_M(Q; distr=:lognormal, σ=1.)
    M = zeros(Q, Q, Q)
    if distr == :lognormal
        p = LogNormal(0., σ)
        for q = 1:Q
            M[:, :, q] .= rand(p, (Q, Q))
            # normalize
            M[:, :, q] ./= sum(M[:, :, q])
        end 
    elseif length(distr) == 3
        S = σ/√2 * randn(Q,Q); S = S + S'
        A = σ/√2 * randn(Q,Q); A = A - A'
        M = √(1-η1^2) * S + η1 * A
        for q = 2:Q
            M[:,:,q] .= η2 * M[:,:,1] + √(1-η2^2) * M[:,:,q]    
        end 
        M = exp.(M)
        # normalize
        for q = 1:Q
            M[:, :, q] ./= sum(M[:, :, q])
        end
        error("not implemented")
    elseif :fixed_ent
        for q = 1:Q
            vals = rand(Q^2)
            ind = rand(1:Q^2)

            vals[ind] += δ
            xlogx.(vals / sum(vals) )/(2*log(Q))
            (1/(sum(vals) +δ) * log(δ/(sum(vals) +δ)) + 1)  
            
            M[:,:,q] 


    end
    return M 
end

function build_tree(M, K; p_root = nothing)
    Q,_,_ = size(M)
    tree = Vector{Int}[]
    if p_root == nothing
        push!(tree, [rand(1:Q)])
    else
        push!(tree, [sample([1:Q;], Weights(p_root))])
    end
    for k = 1:K
        q_list = Int[]
        for j = 1:2^(k-1)
            parent = tree[k][j]
            Mp = vec(M[:, :, parent]) 
            c = sample([1:Q^2;], Weights(Mp))
            left = ((c-1) ÷ Q) + 1; push!(q_list, left) 
            right = ((c-1) % Q) + 1; push!(q_list, right)
        end
        push!(tree, q_list)
    end
    return tree 
end

mutable struct Problem
    # container used to store a sequence "dataset"
    M::Array{Float64, 3} # Q x Q x Q
    trees::Vector{Vector{Vector{Int}}} # S x (K+1) x 2^(k-1) dim
    seqs::Vector{Vector{Int}} # S x 2^(K) dim
    roots::Union{Vector{Int}, Nothing} # S dim
end

function generate_problem(;Q=3, K=4, S=10, σ=1., supervised=false)
    M = generate_M(Q; σ=σ)
    trees = map(s->build_tree(M, K), 1:S)
    seqs = [trees[s][K+1] for s=1:S]
    roots = supervised ? map(s->trees[s][1][1], 1:S) : nothing
    return Problem(M, trees, seqs, roots)
end 

function print_tree(t, letters=true)
    K = length(t)
    l = ""
    for k = 1:K
        for q in t[k]
            letters == true && (q = convert(Char, q + 96))
            l *= " "^(2^((K+1)-k-1)-1)   * "$q" * " "^(2^((K+1)-k-1))
        end
        l *= "\n"
    end
    print(l)
end
function compare_trees(t1, t2, letters=true)
    K = length(t1)
    l = ""
    overlap = zeros(K)
    for k = 1:K
        for (q1,q2) in zip(t1[k],t2[k]) 
            letters == true && (q1 = convert(Char, q1 + 96))
            letters == true && (q2 = convert(Char, q2 + 96))
            l *= " "^(2^((K+1)-k)-2) * "$q1/$q2" * " "^(2^((K+1)-k)-1)
        end
        l *= "\n"
        overlap[K] = sum(t1[k] .== t2[k]) / length(t1)
    end
    print(l)
end

mutable struct Symb 
    # unfortunately the view of the tree is the opposite of the one preferred by Marc. 
    # root (and parents) on top, leaves (and children) on bottom (left and right defined accordingly)
    # note that in this version, each message is a matrix where each column (of dim Q) is associated with one of the S sequences 
    up::Array{Float64, 2} # message (Q x S)
    down::Array{Float64, 2} # message (Q x S)
end

mutable struct Fact
    # each factor is just a set of pointers to the symbols linked to it
    parent::Symb
    right::Symb
    left::Symb
end

mutable struct FactGraph
    # some useful quantities
    Q::Int # letters
    K::Int # tree layers
    S::Int # number of sequences

    # transition tensor
    M::Array{Float64, 3} # Q x Q x Q

    # pointers to the factors and the symbols, organized by layers
    factors::Vector{Vector{Fact}} # K vectors, each containing all the factors in that layer
    symbols::Vector{Vector{Symb}} # (K+1) vectors, each containing all teh symbols in that layer
end

uniform(Q, P) = fill(1/Q, (Q, P))
function build_complete_tree_graph(K, M, S)
    Q,_,_ = size(M)
    factors = Vector{Fact}[]
    symbols = Vector{Symb}[]
    # root
    push!(symbols, [Symb(uniform(Q, S), uniform(Q, S))])
    for k = 1:K
        push!(symbols, [Symb(uniform(Q, S), uniform(Q, S)) for i = 1:2^k])
        f_list = Fact[]
        for j = 1:2^(k-1)
            push!(f_list, Fact(symbols[k][j], symbols[k+1][(j-1)*2+1], symbols[k+1][(j-1)*2+2]))
        end
        push!(factors, f_list)   
    end
    return FactGraph(Q, K, S, M, factors, symbols)
end

function normalize_message!(μ::Array{Float64, 2})
    μ ./= sum(μ, dims=1)
end

function marginals(s::Symb)
    return normalize_message!(s.up .* s.down)
end

function run_BP!(graph::FactGraph)
    @extract graph: K factors M
    # going from leaves to root (up in my notation)
    @inbounds for k = K:-1:1
        for f in factors[k]
            μ_parent, μ_right, μ_left = f.parent.up, f.right.up, f.left.up
            @einsum μ_parent[p, s] = M[l, r, p] * μ_left[l, s] * μ_right[r, s]
            normalize_message!(μ_parent)
        end
    end
    # going from root to leaves (down in my notation)
    @inbounds for k = 1:K
        for f in factors[k]
            μ_parent, μ_right_down, μ_left_down = f.parent.down, f.right.down, f.left.down
            μ_right_up, μ_left_up = f.right.up, f.left.up

            @einsum μ_right_down[r, s] = M[l, r, p] * μ_parent[p, s] * μ_left_up[l, s]
            @einsum μ_left_down[l, s] = M[l, r, p] * μ_parent[p, s] * μ_right_up[r, s]
            normalize_message!(μ_left_down)
            normalize_message!(μ_right_down)
        end
    end
end

function BetheFreeEntropy(graph::FactGraph)
    @extract graph: K S Q factors symbols M
    F = zeros(S)
    for k = 1:K
        for f in factors[k]
            μ_parent, μ_right, μ_left = f.parent.down, f.right.up, f.left.up
            F_f = zeros(S)
            @einsum F_f[s] = M[l, r, p] * μ_left[l, s] * μ_right[r, s] * μ_parent[p, s]
            F .+= log.(F_f)
        end
        k == 1 && continue
        for s in symbols[k]
            F_s = sum(s.up .* s.down, dims=1)
            F .-= log.(F_s)'
        end
    end
    return -F / (log(Q) * 2^K)
end

function dM_BetheFreeEntropy(graph::FactGraph)
    @extract graph: K Q S factors symbols M
    dM = zeros(Q, Q, Q)
    dM_f = zeros(Q, Q, Q)
    for k = 1:K
        for f in factors[k]
            μ_parent, μ_right, μ_left = f.parent.down, f.right.up, f.left.up
            F_f = zeros(S)
            dM_f .= 0. 
            @einsum F_f[s] = M[l, r, p] * μ_left[l, s] * μ_right[r, s] * μ_parent[p, s]
            @einsum dM_f[l, r, p] = μ_left[l, s] * μ_right[r, s] * μ_parent[p, s] / F_f[s]
            dM .+= dM_f
        end
    end
    return dM / (log(Q) * 2^K * S)
end

function fix!(s_mess::Array{Float64, 2}, qs::Vector{Int}) # fix up or down message of some degree-1 variable to a delta on state V
    fill!(s_mess, 0.)
    S = length(qs)
    @inbounds for s = 1:S
        s_mess[qs[s], s] = 1.
    end
end

function run_BP_on_seqs!(graph::FactGraph, seqs::Vector{Vector{Int}}, roots::Union{Vector{Int}, Nothing})
    @extract graph: K Q S M symbols
    @assert length(seqs) == S

    if roots != nothing 
        # in the supervised setting fix the root
        fix!(symbols[1][1].down, roots)
    else
        # otherwise set to uniform
        symbols[1][1].down .= uniform(Q, S)
    end
    
    s_boundary = symbols[K+1]
    for i = 1:length(seqs[1])
        # fix the leaves according to the sequences 
        fix!(s_boundary[i].up, map(s->seqs[s][i], 1:S))
    end
    run_BP!(graph)
    return
end

function MAP_trees(graph)
    # obtains the argmax of the marginals in he format of a tree 
    @extract graph: K S M symbols
    Q,_,_ = size(M)
    trees = [Vector{Int}[] for s=1:S]
    likelihoods = zeros(S)
    for k = 1:K+1
        lists = [Int[] for s=1:S]
        for s in symbols[k]
            probs, inds = findmax(marginals(s), dims=1)
            for s = 1:S
                likelihoods .+= probs'
                push!(lists[s], inds[s][1])
            end
        end    
        for s = 1:S
            push!(trees[s], lists[s])
        end
    end
    return trees, likelihoods
end

function overlap_trees(true_trees, rec_trees)
    # to compute overlap btw true tree and MAP tree
    S = length(true_trees)
    K = length(true_trees[1])-1
    correct_fraction = zeros(K+1)
    for k = 1:K+1
        l = length(rec_trees[1][k])
        for s = 1:S
            correct_fraction[k] += sum(true_trees[s][k] .== rec_trees[s][k]) / l
        end
        correct_fraction[k] /= S
        println("layer $(k-1), correct fraction: ", correct_fraction[k])
    end
end

function trees_likelihood(graph, tree)
    # computes the "likelihood" according to the BP marginals of any given tree (assuming a given sequence is fixed at the leaves)
    @extract graph: S K symbols
    margs = [marginals.(symbols[k]) for k = 1:K] 
    likelihoods = zeros(S)
    for k = 1:K
        for s = 1:S 
            tree_k = tree[s][k]
            margs_k = margs[k]
            for i = 1:length(tree_k)
                likelihoods[s] += margs_k[i][tree_k[i]]
            end
        end
    end
    return likelihoods
end

function KL(M1, M2)
    Q,_,_ = size(M1)
    @assert size(M2)[1] == Q 
    kl = 0.
    for q = 1:Q 
        kl += sum(xlog2x.(M1) - M1 .* log2.(M2 .+ 1e-16))
    end 
    return kl
end 

function tree_reconstruction(problem; η=0.05, λ=0.1, T=100, infotime=1, print_trees=false, ϵ=1e-3, verb=1)
    @extract problem: M seqs roots
    S = length(seqs)
    Q = size(M)[1]
    K = round(Int, log2(length(seqs[1])))
    
    M_guess = generate_M(Q; σ=0.) # initial guess
    graph = build_complete_tree_graph(K, M_guess, S) # initializes factor graph
    
    # evaluate the initial condition
    if verb > 0
        run_BP_on_seqs!(graph, seqs, roots)
        rec_MSE = sum(abs2, M_guess - M) / Q
        rec_KL = KL(M, M_guess)
        graph_entropy = mean(BetheFreeEntropy(graph))
        M_entropy = -sum(xlogx.(M_guess)) / (2 * Q * log(Q))
        println("initial rec_MSE=", rec_MSE, " rec_KL=", rec_KL, " entropy=", graph_entropy, " Mentropy=", M_entropy)
        rec_trees, rec_likelihoods = MAP_trees(graph)
        avg_overlap_trees = overlap_trees(problem.trees, rec_trees)
        println()
    end
    dM = 0. # ugly but otherwise this variable is not defined
    for t = 1:T
        if verb>0 && t % infotime == 0 # when we print some info
            rec_MSE = sum(abs2, M_guess - M) / Q
            rec_KL = KL(M, M_guess)
            graph_entropy = mean(BetheFreeEntropy(graph))
            M_entropy = -sum(xlogx.(M_guess)) / (2 * Q * log(Q))
            println("t=$t |dM|=$(sum(abs2, dM)/ Q) rec_MSE=", rec_MSE, " rec_KL=", rec_KL, " entropy=", graph_entropy, " Mentropy=", M_entropy)
            rec_trees, rec_likelihoods = MAP_trees(graph)
            avg_overlap_trees = overlap_trees(problem.trees, rec_trees)
            println()   
        end

        # expectation step  
        run_BP_on_seqs!(graph, seqs, roots) 
        
        # maximization step
        dM = dM_BetheFreeEntropy(graph) 
        for q = 1:Q
            dM[:,:,q] .-= sum(dM[:,:,q]) / Q^2 # this ensures that tot prob = 1 is preserved by the update
        end
        M_guess .+= η * dM
        clamp!(M_guess, 0., 1) # this ensures that M elements do not become negative
        for q = 1:Q
            M_guess[:,:,q] ./= sum(M_guess[:,:,q])
        end
        graph.M .= M_guess
        
        # stopping criterion
        if (sum(abs2, dM)/ Q) < ϵ
            verb>0 && println("converged!")
            break 
        end
    end
    rec_MSE = sum(abs2, M_guess - M) / Q
    rec_KL = KL(M, M_guess)
    verb>0 && println("final rec_MSE=", rec_MSE, " rec_KL=", rec_KL)
    # compute most likely trees according to BP marginals
    verb>0 && (rec_trees, rec_likelihoods = MAP_trees(graph))
    # compute avg overlap btw true and reconstructed trees 
    verb > 0 && (avg_overlap_trees = overlap_trees(problem.trees, rec_trees))
       
    # return M_guess, graph
    return rec_MSE, rec_KL
end

function test1(;
    resfile = "res_M_inference.txt",
    seeds = 10
    )
    if !isfile(resfile)  
        f = open(resfile, "w")
        println(f, "# 1:Q 2:σ 3:K 4:S 5:supervised 6:μ_MSE 7:σ_MSE 8:μ_KL 9:σ_KL")
        close(f)
    end
    for Q in [2,3], σ in [0.5,1.,2.,4.], K in [10], S in [5, 10, 20, 40, 80], supervised in [true, false]
        println("doing: Q=$Q, σ=$σ, K=$K, S=$S, sup=$supervised")
        MSEs=Float64[]; KLs=Float64[];
        @time for seed = 1:seeds
            print(".")
            prob = generate_problem(Q=Q,K=K,S=S, σ=σ, supervised=supervised)
            MSE, KL = tree_reconstruction(prob; η=0.5e-2, T=10000, verb=0, infotime=100000, ϵ=1e-8)
            push!(MSEs, MSE); push!(KLs, KL);
        end
        μ_KL = mean(KLs); σ_KL= std(KLs)
        μ_MSE = mean(MSEs); σ_MSE= std(MSEs)
        f = open(resfile, "a")
        println("MSE=(", μ_MSE, ",", σ_MSE, ") KL=(", μ_KL, ",", σ_KL, ")")
        println()
        println(f, Q, " ", σ, " ", K, " ", S, " ", supervised, " ", μ_MSE, " ", σ_MSE, " ", μ_KL, " ", σ_KL)
        close(f)
    end 
end