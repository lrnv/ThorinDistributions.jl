struct PreComp{Tb,Tl,Tf,Tm}
    BINS::Tb
    LAGUERRE::Tl
    FACTS::Tf
    MAX_SUM_OF_M::Tm
end

"""
    PreComp(m)

Given a precison of computations and a tuple m which gives the size of the future laguerre basis, this fonctions precomputes certain quatities
these quatities might be needed later...
"""
function PreComp(m)
    setprecision(1024)
    m = big(m)
    BINS = zeros(BigInt,(m,m))
    FACTS = zeros(BigInt,m)
    LAGUERRE = zeros(BigFloat,(m,m))
    for i in 1:m
        FACTS[i] = factorial(i-1)
    end
    for i in 1:m, j in 1:m
        BINS[i,j] = binomial(i-1,j-1)
        LAGUERRE[i,j] = -BINS[i,j]/FACTS[j]*(-big(2))^(j-1)
    end
    T = Double64
    BINS = T.(BINS)
    LAGUERRE = T.(LAGUERRE)
    FACTS = T.(FACTS)
    PreComp(BINS,LAGUERRE,FACTS,m)
end

const P = PreComp(100)

"""
    get_coefficients(α,θ,m)

α should be a vector of shapes of length n, θ should be a matrix of θ of size (n,d) for a MultivariateGammaConvolution with d marginals.

This function produce the coefficients of the multivariate (tensorised) expensions of the density. It is type stable and quite optimized.
it is based on some precomputations that are done in a certain precision.

"""
function get_coefficients(α, θ, m)
    # α must be an array with size (n,)
    # θ must be an array with size (n,d)
    # max_p must be a Tuple of integers with size (d,)
    T = Double64
    α = T.(α)
    θ = T.(θ)

    # Trim down null αs values:
    are_pos = α .!= 0
    θ = θ[are_pos,:]
    α = α[are_pos]

    # Allocates ressources, construct the simplex expression of the θ and the indexes.
    coefs = zeros(T,m)
    κ = deepcopy(coefs)
    μ = deepcopy(coefs)
    n = size(θ)[1]
    d = ndims(coefs)
    I = CartesianIndices(coefs)
    na = [CartesianIndex()]
    S = θ ./ (T(1) .+ sum(θ,dims=2))
    # all S must be smaller than one, and sum maximum to 1
    for i in 1:size(S,2)
        if sum(S[:,i]) > T(1)
            S[:,i] .-= 2*eps(T)
        end
    end
    S_pow = S[na,:,:] .^ (0:Base.maximum(m))[:,na,na]

    # Edge case for the Oth cumulant, 0th moment and 0th coef:
    # this log sometimes fails, when sum(S,dims=2) is greater than 1. For this, we might add a restriction here.
    κ[1] = sum(α .* log.(T(1) .- sum(S,dims=2)))
    coefs[1] = μ[1] = exp(κ[1])

    for k in I[2:length(I)]
        # Indices and organisation
        k_arr = [ki for ki in Tuple(k)]
        degree = findfirst(k_arr .!= 1)
        sk = sum(k_arr)

        # Main Computations:
        for i in 1:n
            rez = α[i]
            for j in 1:d
                rez *= S_pow[k[j],i,j]
            end
            κ[k] += rez
        end
        κ[k] *= P.FACTS[sk-d]

        for j in CartesianIndices(k)
            add_mu = j[degree] < k[degree]
            if add_mu
                rez_mu = μ[j] * κ[k - j + I[1]]
            end
            rez_coefs = μ[j]
            for i in 1:d

                if add_mu
                    rez_mu *= i==degree ? P.BINS[k[i]-1, j[i]] : P.BINS[k[i], j[i]]
                end
                rez_coefs *= P.LAGUERRE[k[i], j[i]]
            end
            if add_mu
                μ[k] += rez_mu
            end
            coefs[k] += rez_coefs
        end
    end
    coefs *= sqrt(T(2))^d
    return coefs
end


"""
    laguerre_L_2x(x,p)

Compute univariate laguerre polynomials Lₚ(2x).
"""
function laguerre_L_2x(x,p)
    sum(-P.LAGUERRE[p+1,1:(p+1)] .* x .^ (0:p))
end

"""
    laguerre_phi(x,p)

Computes the function ϕₚ(x) = √2 e⁻ˣ Lₚ(2x). This functions form an orthonormal basis of R₊
"""
function laguerre_phi(x,p)
    exp(-x) * sqrt(2) * laguerre_L_2x(x,p)
end

"""
    laguerre_density(x,coefs)

Given some laguerre coefficients, computes the correpsonding function at the point x.
"""
function laguerre_density(x,coefs)
    rez = zero(Base.eltype(coefs))
    for p in CartesianIndices(coefs)
        rez += coefs[p] * prod(laguerre_phi.(x,Tuple(p) .-1))
    end
    return max(rez,0)
end


"""
    laguerre_phi_several_pts(x,max_p)

Computes the laguerre_phi for each row of x and each p in CartesianIndex(maxp)
This is a lot more efficient than broadcasting the laguerre_phi function,
but this is ugly in the sense that we re-wrote some of the code.
"""
function laguerre_phi_several_pts(x,max_p)
    # computes the laguerre_phi for each row of x and each p in CartesianIndex(maxp)
    # This is a lot more efficient than broadcasting the laguerre_phi function,
    # but this is ugly in the sens that we re-wrote some of the code.


    d,n = size(x)
    rez = ones(Base.eltype(x),(n,max_p...))
    na = [CartesianIndex()]
    MP = Base.maximum(max_p)

    println("Computing laguerre_L")
    laguerre_L = zeros(Base.eltype(x),(d,MP,n))
    powers = x[:,na,:] .^ (0:(MP-1))[na,:,na]
    Threads.@threads for p in 1:MP
        laguerre_L[:,p:p,:] = dropdims(sum(-P.LAGUERRE[p:p,1:MP][na,:,:,na] .* powers[:,na,:,:],dims=3),dims=3)
    end


    #laguerre_L = dropdims(sum(-P.LAGUERRE[1:MP,1:MP][na,:,:,na] .* powers[:,na,:,:],dims=3),dims=3) # (d,Mp,n)

    println("Computing exponentials...")
    exponentials = dropdims(exp.(-sum(x,dims=1)),dims=1)

    println("Computing L(2x)")
    Threads.@threads for p in CartesianIndices(max_p)
        for i in 1:d
            rez[[CartesianIndex((i,Tuple(p)...)) for i in 1:n]] .*= laguerre_L[i,p[i],:]
        end
        print(p,"\n")
    end
    return rez .* sqrt(big(2))^d .* exponentials
end

"""
    empirical_coefs(x,maxp)

Compute the empirical laguerre coefficients of the density of the random vector x, given as a Matrix with n row (number of samples) and d columns (number of dimensions)
"""
function empirical_coefs(x,maxp)
    x = Double64.(x)
    return dropdims(sum(laguerre_phi_several_pts(x,maxp),dims=1)/last(size(x)),dims=1)
end

"""
    L2Objective(par,emp_coefs)

A L2 distance to be minimized between laguerre coefficients of MultivariateGammaConvolutions and empirical laguerre coefficients.
This distance is some kind of very high degree polynomial * exponentials, so minimizing it is very hard.
"""
function L2Objective(par, emp_coefs)
    m = size(emp_coefs)
    d = length(m)
    n = Int((size(par)[1])/(d+1))
    α = par[1:n] .^2 #make them positives
    rates = reshape(par[(n+1):(d+1)*n],(n,d)) .^ 2 # make them positives
    coefs = get_coefficients(α,rates,m)
    return sum((coefs .- emp_coefs) .^2)
end

"""
    L2ObjectiveWithPenalty(par,emp_coefs)

A L2 distance to be minimized between laguerre coefficients of MultivariateGammaConvolutions and empirical laguerre coefficients.
This distance is some kind of very high degree polynomial * exponentials, so minimizing it is very hard.
This version includes a penalty to force parameters to go towards 0. but it is not yet working correctly.
"""
function L2ObjectiveWithPenalty(par,emp_coefs)
    old_par = par[1:(length(par)-1)]
    loss = L2Objective(old_par,emp_coefs)
    lambda = last(par)^2
    penalty = lambda * sum(abs.(old_par))
    return loss + penalty
end


function get_uniform_x0(n,d)
    θ = zeros(BigFloat, (n,d+1))
    Random.rand!(θ)
    θ = -log.(θ) # exponentials
    θ ./= sum(θ, dims=2) # rowsums = 1
    θ = θ[:,1:d] # uniform on the simplex.
    θ ./= (big(1.0) .- sum(θ,dims=2))
    θ = sqrt.(θ)

    αs = zeros(BigFloat,(n,))
    Random.rand!(αs)
    αs = - log.(αs)

    # Finaly, merge the two:
    par = vcat(αs,reshape(θ,(n*d)))
    return par
end

"""
    minimum_m(n,d)

Computes the minimum integer m such that m^d > (d+1)n
"""
function minimum_m(n,d)
    return Int(ceil(((d+1)n)^(1/d)))
end
