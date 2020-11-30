
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
    setprecision(2048)
    m = big(m)
    BINS = zeros(BigInt,(m,m))
    FACTS = zeros(BigInt,m)
    LAGUERRE = zeros(BigFloat,(m,m))
    for i in 1:m
        FACTS[i] = factorial(i-1)
    end
    for i in 1:m, j in 1:m
        BINS[j,i] = binomial(i-1,j-1)
        LAGUERRE[j,i] = BINS[j,i]/FACTS[j]*(-big(2))^(j-1)
    end
    BINS = ArbT.(BINS)
    LAGUERRE = ArbT.(LAGUERRE)
    FACTS = ArbT.(FACTS)
    PreComp(BINS,LAGUERRE,FACTS,ArbT(m))
end
const P = PreComp(200)

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
    entry_type = Base.promote_eltype(α,θ,[1.0]) # At least float.
    α = ArbT.(α)
    θ = ArbT.(θ)

    # Trim down null αs values:
    are_pos = α .!= 0
    θ = θ[are_pos,:]
    α = α[are_pos]

    coefs = zeros(ArbT,m)
    build_coefficients!(coefs,α,θ,ArbT(1.0),m)
    return entry_type.(coefs)
end

function build_coefficients!(coefs,α,θ,cst1,m)
    # Allocates ressources
    κ = deepcopy(coefs)
    μ = deepcopy(coefs)
    n = size(θ)[1]
    d = length(m)
    I = CartesianIndices(coefs)
    S = θ ./ (cst1 .+ sum(θ,dims=2))
    S_pow = [s^k for k in (0:Base.maximum(m)), s in S]

    # Starting the algorithm: there is an edge case for the Oth cumulant, 0th moment and 0th coef:
    κ[1] = sum(α .* log.(cst1 .- sum(S,dims=2))) # this log fails ifsum(S,dims=2) is greater than 1, which should not happend.
    μ[1] = exp(κ[1])
    coefs[1] = μ[1]

    @inbounds for k in I[2:length(I)]
        # Indices and organisation
        k_arr = Tuple(k)
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
            rez_coefs = μ[j]
            if j[degree] < k[degree]
                rez_mu = μ[j] * κ[k - j + I[1]]
                for i in 1:d
                    rez_mu *= P.BINS[j[i],k[i]-Int(i==degree)]
                    rez_coefs *= P.LAGUERRE[j[i], k[i]]
                end
                μ[k] += rez_mu
            else
                for i in 1:d
                    rez_coefs *= P.LAGUERRE[j[i], k[i]]
                end
            end
            coefs[k] += rez_coefs
        end
    end
    coefs .*= sqrt(2*cst1)^d
end

function build_coefficients!(coefs,α,θ,cst1,m::NTuple{1, T}) where T <: Int

    θ = θ[:,1]
    m = m[1]
    kappa = Array{eltype(α), 1}(undef, m)
    mu = zeros(eltype(α), m)

    @. θ /= 1 + θ

    # Edge case for the Oth cumulant, 0th moment and 0th coef:
    kappa[1] = sum(α .* log.(cst1 .- θ))
    coefs[1] = mu[1] = exp(kappa[1])

    @inbounds for k in 2:m
        # Main Computations:
        α .*= θ
        kappa[k] = sum(α) * P.FACTS[k - 1]
    
        for j in 1:k-1
            mu[k] += mu[j] * kappa[k - j + 1] * P.BINS[j, k-1]
        end
        
        @views coefs[k] = sum(mu[1:k] .* P.LAGUERRE[1:k, k])
    end

    coefs .*= sqrt(2*cst1)
end

function build_coefficients!(coefs,α,θ,cst1,m::NTuple{2, T}) where T <: Int
    # Allocates ressources
    κ = Array{Base.eltype(α), 2}(undef, m)
    μ = zeros(Base.eltype(α), m)

    #n = size(θ)[1]
    d = length(m)
    I = CartesianIndices(coefs)
    θ ./= (cst1 .+ sum(θ,dims=2))
    S_pow = [s^k for k in (0:Base.maximum(m)), s in θ]

    # reduce the number of multiplications : 
    for i in 1:size(α,1)
        S_pow[:,i,:] .*= sqrt(α[i])
    end

    # Starting the algorithm: there is an edge case for the Oth cumulant, 0th moment and 0th coef:
    κ[1] = sum(α .* log.(cst1 .- sum(θ,dims=2))) # this log fails ifsum(S,dims=2) is greater than 1, which should not happend.
    μ[1] = exp(κ[1])
    coefs[1] = μ[1]

    @inbounds for k in I[2:length(I)]
        # Indices and organisation
        degree = k[1]!=1 ? 1 : 2

        κ[k] = sum(S_pow[k[1],:,1] .* S_pow[k[2],:,2]) * P.FACTS[sum(Tuple(k))-d]
        #κ[k] = sum(S_pow[k[1],:,1] .* S_pow[k[2],:,2]) * P.FACTS[sum(Tuple(k))-d] / P.FACTS[k[1]] / P.FACTS[k[2]]
        for j in CartesianIndices(k)
            if j[degree] < k[degree]
                #μ[k] += μ[j] * κ[k - j + I[1]] * (k[degree] - j[degree] +1)
                μ[k] += μ[j] * κ[k - j + I[1]] *  P.BINS[j[1],k[1]-Int(1==degree)] * P.BINS[j[2],k[2]-Int(2==degree)]
            end
            coefs[k] += μ[j] * P.LAGUERRE[j[1], k[1]] * P.LAGUERRE[j[2], k[2]]
        end
    end
    coefs .*= sqrt(2*cst1)^d
end


"""
    laguerre_L_2x(x,p)

Compute univariate laguerre polynomials Lₚ(2x).
"""
function laguerre_L_2x(x,p)
    sum(P.LAGUERRE[1:(p+1),p+1] .* x .^ (0:p))
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
    rez = ones(ArbT,(n,max_p...))
    na = [CartesianIndex()]
    MP = Base.maximum(max_p)

    println("Computing laguerre_L")
    laguerre_L = zeros(ArbT,(d,MP,n))
    powers = x[:,na,:] .^ (0:(MP-1))[na,:,na]
    Threads.@threads for p in 1:MP
        laguerre_L[:,p:p,:] = dropdims(sum(P.LAGUERRE[1:MP,p:p][na,:,:,na] .* powers[:,:,na,:],dims=2),dims=2)
    end

    println("Computing exponentials...")
    exponentials = dropdims(exp.(-sum(x,dims=1)),dims=1)

    println("Computing L(2x)")
    Threads.@threads for p in CartesianIndices(max_p)
        for i in 1:d
            rez[[CartesianIndex((i,Tuple(p)...)) for i in 1:n]] .*= laguerre_L[i,p[i],:]
        end
        print(p,"\n")
    end
    return rez .* sqrt(ArbT(2))^d .* exponentials
end

"""
    empirical_coefs(x,maxp)

Compute the empirical laguerre coefficients of the density of the random vector x, given as a Matrix with n row (number of samples) and d columns (number of dimensions)
"""
function empirical_coefs(x,maxp)
    entry_type = Base.promote_eltype(x,[1.0])
    x = ArbT.(x)
    y = laguerre_phi_several_pts(x,maxp)
    return entry_type.(dropdims(sum(y,dims=1)/size(y,1),dims=1))
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


# function get_uniform_x0{T}(n,d) where T
#     θ = zeros(T, (n,d+1))
#     Random.rand!(θ)
#     θ = -log.(θ) # exponentials
#     θ ./= sum(θ, dims=2) # rowsums = 1
#     θ = θ[:,1:d] # uniform on the simplex.
#     θ ./= (T(1) .- sum(θ,dims=2))
#     θ = sqrt.(θ)
#
#     αs = zeros(T,(n,))
#     Random.rand!(αs)
#     αs = - log.(αs)
#
#     # Finaly, merge the two:
#     par = vcat(αs,reshape(θ,(n*d)))
#     return par
# end

"""
    minimum_m(n,d)

Computes the minimum integer m such that m^d > (d+1)n
"""
function minimum_m(n,d)
    return Int(ceil(((d+1)n)^(1/d)))
end
