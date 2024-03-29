"""
    get_coefficients(α,θ,m)

α should be a vector of shapes of length n, θ should be a matrix of θ of size (n,d) for a MultivariateGammaConvolution with d marginals.

This function produce the coefficients of the multivariate (tensorised) expensions of the density. It is type stable and quite optimized.
it is based on some precomputations that are done in a certain precision.

"""
function get_coefficients(α, θ, m; P = nothing)
    # α must be an array with size (n,)
    # θ must be an array with size (n,d)
    # max_p must be a Tuple of integers with size (d,)
    T = Base.promote_eltype(α,θ,[1.0]) # At least float.
    α = T.(α)
    θ = T.(θ)

    # Trim down null αs values:
    are_pos = findall(α .> eps(T))
    if ndims(θ) == 1
        θ = θ[are_pos]
    else
        θ = θ[are_pos,:] # produces a matrix when θ is only a vector.. 
    end
    α = α[are_pos]
    if isnothing(P)
        P = get_precomp(T,sum(m))::PreComp{T}
    end
    coefs = zeros(T,m)
    build_coefficients!(coefs,α,θ,T(1.0),m,P)
    return coefs
end

function build_coefficients!(coefs,α,θ,cst1,m,P)

    # This is the default method, for higher dimensions.

    # Allocates ressources
    κ = deepcopy(coefs)
    μ = deepcopy(coefs)
    n = size(θ)[1]
    d = length(m)
    I = CartesianIndices(coefs)
    S = θ ./ (cst1 .+ sum(θ,dims=2))
    S_pow = [s^k for k in (0:Base.maximum(m)), s in S]

    # Starting the algorithm: there is an edge case for the Oth cumulant, 0th moment and 0th coef:
    to_log = cst1 .- sum(S,dims=2)
    to_log .= ifelse.(to_log .< 0, to_log .+ eps(Base.eltype(α)), to_log)
    κ[1] = sum(α .* log.(to_log))
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

function build_coefficients!(coefs::AbstractArray{T,1},α::AbstractArray{T,1},θ::AbstractArray{T,1},cst1::T,m::NTuple{1, T2},P::PreComp{T}) where {T2 <: Int,T <: Real}

    #kappa = Array{Base.eltype(α), 1}(undef, m[1])
    kappa = zeros(T,m[1])
    mu = zeros(T, m[1])

    @. θ /= 1 + θ

    # Edge case for the Oth cumulant, 0th moment and 0th coef:
    to_log = cst1 .- θ
    to_log .= ifelse.(to_log .< 0, to_log .+ eps(Base.eltype(α)), to_log) # correcting rounding errors.
    kappa[1] = sum(α .* log.(to_log))
    coefs[1] = mu[1] = exp(kappa[1]) # this is the only place kappa[1] is used. 
    #coefs[1] = mu[1] = prod(to_log .^ α)
    @inbounds for k in 2:m[1]
        # Main Computations:
        α .*= θ
        kappa[k] = sum(α)
        @inbounds for j in 1:k-1
            mu[k] += mu[j] * kappa[k-j+1]
        end
        mu[k] /= k-1
        @views coefs[k] = P.LAGUERRE2[1:k, k]'mu[1:k]
    end
end

function build_coefficients!(coefs::AbstractArray{T,1},α::AbstractArray{T,1},θ::AbstractArray{T,2},cst1::T,m::NTuple{2, T2},P::PreComp{T}) where {T2 <: Int,T <: Real}
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
    to_log = cst1 .- sum(θ,dims=2)
    to_log .= ifelse.(to_log .< 0, to_log .+ eps(Base.eltype(α)), to_log)
    κ[1] = sum(α .* log.(to_log)) # this log fails ifsum(S,dims=2) is greater than 1, which occurs because of rounding errors. We add an eps to avoid that. 
    μ[1] = exp(κ[1])
    coefs[1] = μ[1]

    @inbounds for k in I[2:length(I)]
        # Indices and organisation
        degree = k[1]!=1 ? 1 : 2
        deg1 = Int(1==degree)
        deg2 = Int(2==degree)
        
        κ[k] = sum(S_pow[k[1],:,1] .* S_pow[k[2],:,2]) * P.FACTS[sum(Tuple(k))-d]
        for j in CartesianIndices(k)
            if j[degree] < k[degree]
                μ[k] += μ[j] * κ[k - j + I[1]] *  P.BINS[j[1],k[1]-deg1] * P.BINS[j[2],k[2]-deg2]
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
    P = get_precomp(Base.eltype(x),p+1)
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
but this is ugly in the sense that we re-wrote some of the code. This machanisme could be re-written more efficiently.
"""
function laguerre_phi_several_pts(x::AbstractArray{T},max_p) where {T <: Real}
    P = get_precomp(T,Base.maximum(max_p)+1)::PreComp{T}
    d,n = size(x) 
    rez = ones(T,(n,max_p...))
    MP = Base.maximum(max_p)

    # println("Computing exponentials...")
    exponentials = dropdims(exp.(-sum(x,dims=1)),dims=1)

    laguerre_L = zeros(T,(d,MP,n))
    # println("Computing powers...")
    powers = x[:,na,:] .^ (0:(MP-1))[na,:,na]
    # println("Computing laguerre_L")
    Threads.@threads for p in 1:MP
        laguerre_L[:,p:p,:] = dropdims(sum(P.LAGUERRE[1:p,p:p][na,:,:,na] .* powers[:,1:p,na,:],dims=2),dims=2)
        print(p,"\n")
    end

    # println("Computing L(2x)")
    Threads.@threads for p in CartesianIndices(max_p)
        for i in 1:d
            rez[[CartesianIndex((i,Tuple(p)...)) for i in 1:n]] .*= laguerre_L[i,p[i],:]
        end
        print(p,"\n")
    end
    return rez .* sqrt(T(2))^d .* exponentials
end

"""
    empirical_coefs(x,maxp)

Efficiently Compute the empirical laguerre coefficients of the density of the random vector x, given as a Matrix with n row (number of samples) and d columns (number of dimensions)
"""
function empirical_coefs(x,max_p)
    T = Base.promote_eltype(x,[1.0])
    x = T.(x)
    y = laguerre_phi_several_pts(x,max_p)
    # println("Taking the means...")
    return dropdims(sum(y,dims=1)/size(y,1),dims=1)
end


function empirical_coefs(x,max_p::NTuple{2,M}) where M<:Int
    T = Base.promote_eltype(x,[1.0])
    x = T.(x)
    P = get_precomp(T,Base.maximum(max_p)+1)::PreComp{T}
    d,n = size(x) 
    rez = ones(T,max_p)
    MP = Base.maximum(max_p)

    exponentials = dropdims(exp.(-sum(x,dims=1)),dims=1)

    laguerre_L = zeros(T,(d,MP,n))
    powers = x[:,na,:] .^ (0:(MP-1))[na,:,na]
    Threads.@threads for p in 1:MP
        laguerre_L[:,p:p,:] = dropdims(sum(P.LAGUERRE[1:p,p:p][na,:,:,na] .* powers[:,1:p,na,:],dims=2),dims=2)
    end

    Threads.@threads for p in CartesianIndices(max_p)
        rez[p] = sum(laguerre_L[1,p[1],:] .* laguerre_L[2,p[2],:] .* exponentials)
    end
    rez = (rez .* 2) ./ n
    return rez
end


"""
    L2Objective(par,emp_coefs)

A L2 distance to be minimized between laguerre coefficients of MultivariateGammaConvolutions and empirical laguerre coefficients.
This distance is some kind of very high degree polynomial * exponentials, so minimizing it is very hard.
We recomand using some global minimisation routine, we found the Particle Swarm optimizers from Optim.jl particularly efficient.
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

function αθ_from_par(par)
    n = Int(length(par)/2)
    return par[1:n].^2, par[(n+1):2n].^2
end

