# The goal of this file is to compute the k_statistics polynomials as given by 
# https://discourse.julialang.org/t/help-needed-translate-10-lines-from-maple-to-julia/60650
# and 
# https://iris.unito.it/retrieve/handle/2318/1561388/399488/FASTpolykaysMAPLE.pdf
# also see 
# Di Nardo E., G. Guarino, D. Senato (2007), A new method for fast computing unbiased estimators of cumulants. In press Statistics and Computing. http://www.springer.com/statistics/computational/journal/11222 (download from http://www.unibas.it/utenti/dinardo/lavori.html)

# We also compute their variances, but thoughr bootstrap, although there is probably a closed form for them. 


function _better_stirling(n::T,k::T) where T<:Integer
# Algorithm taken directly from Combinatorics.jl, but we compute the whole array instead of only one by one. 
arr = zeros(T,(n,k))
@inbounds for ni in 1:n
    @inbounds for ki in 1:min(ni,k)
        if ki ∈ (1,ni)
            arr[ni,ki] = 1
        elseif ki == ni - 1
            arr[ni,ki] = binomial(ni, 2)
        elseif ki == 2
            arr[ni,ki] = 2^(ni-1) - 1
        else
            @views arr[ni,ki] = ki * arr[ni - 1, ki] + arr[ni - 1, ki- 1]
        end
    end
end
return arr
end
# 

_fd(k,j) = Polynomials.fromroots([i+j for i in 0:(k-j)])

function _compute_k(N,S)
m = BigInt(length(S))
result = []
stir = _better_stirling(m,m)
_facts = factorial.(0:m)
alternate__facts = (-1) .^ (0:m) .* _facts
u = Array{Polynomials.Polynomial{BigInt}}(undef,(m,))

for k in 1:m
    part = Combinatorics.integer_partitions(k)  
    u[k] = Polynomials.Polynomial([0,(stir[k,1:k] .* alternate__facts[1:k])...])
    u2 = alternate__facts[1:k] .* _fd.(k-1,1:k)
    num = 0
    denom = prod(N .- (0:(k-1)))
    
    for i in eachindex(part)
        Cst = BigInt(_facts[k+1]//prod(_facts[v+1]^n * _facts[n+1] for (v,n) in StatsBase.countmap(part[i])))
        Ns = (prod(u[part[i]]).coeffs[2:end]'u2)(N)
        Ss = prod(S[part[i]])
        num += Cst * Ns * Ss 
    end
    
    push!(result,num/denom)
end
return result
end
# # As a simple check, we can verify that the polynomials holds versus the theoretical ones. 

# # More examples could be found in Fisher's work. 1929
struct FastPols
    N::Int64
    m::Int64
    pols::Vector{StaticPolynomials.Polynomial{BigFloat, SE, Nothing} where SE}
end

function FastPols(N,m)
    DynamicPolynomials.@polyvar S[1:m]
    pols = _compute_k(N,S)
    faster = StaticPolynomials.Polynomial.(pols)
    return FastPols(N,m,faster)
end

function _check_n_m!(fs::FastPols,N,m)
    if isnothing(fs)
        fs = FastPols(n,m)
    else
        @assert fs.N ==N
        @assert fs.m >= m
    end
end

# function k_statistics(data::Vector{T},t_star,m; k_stats_funs::FastPols) where T

#     # Empirical verxion from data 
#     n = length(data) 
#     ET_weights = exp.(t_star .* data) ./ n
#     if isnothing(k_stats_funs)
#         k_stats_funs = FastPols(n,m)
#     else
#         _check_n_m(k_stats_funs,n,m)
#     end
#     sw = sum(ET_weights)
#     zeroth_cumulant = log(sw)
#     ET_weights ./= sw
#     ET_S = vec(sum(data .^ (1:m)' .* ET_weights,dims=1))
#     ET_k_stats = [k_stats_funs.pols[i](ET_S)::T for i in 1:m]::Vector{T}
#     ET_k_stats ./= factorial.(0:(m-1))
#     return [zeroth_cumulant, ET_k_stats...]
# end

function k_statistics(data::Vector{T},t_star,m; k_stats_funs::FastPols) where T
    # Empirical verxion from data 
    n = length(data) 
    wts = exp.(t_star .* data) ./ n
    _check_n_m!(k_stats_funs,n,m)

    k0 = log(sum(wts))
    wts ./= sum(wts) 
    S = vec(sum(data .^ (1:m)' .* wts,dims=1))*n
    k = [k_stats_funs.pols[i](S)::T for i in 1:m]::Vector{T}
    k ./= factorial.(0:(m-1))
    return [k0, k...]
end

function resampled_k_statistics(M,D::Vector{T},t_star,m; k_stats_funs::FastPols) where T
    # first step, resample the dataset D:
    n = length(D) 
    _check_n_m!(k_stats_funs,n,m)

    ks = Array{T}(undef,(M,m+1))
    facts = factorial.(big.(0:(m-1)))
    arr = exp.(t_star .* D) .* (D.^(0:m)')
    arr[:,1] ./= n
    Threads.@threads for i in 1:M 
        println(i)
        ks[i,:] = sum(arr[StatsBase.sample(1:n,n,replace=true),:], dims=1)
        ks[i,2:end] ./= ks[i,1] # Added line to match the non-bootstrapped k-staistics.
        ks[i,1] = log(ks[i,1])
        # the choice of the facts noramlisation highly condition the weights matrix. 
        ks[i,2:end] = map(f->f(ks[i,2:end])::T,k_stats_funs.pols) ./ facts #or .* facts, which gave better fitting results, but is wrong...
    end
    return ks
end

function ugc_k_stats(α::Vector{T},θ::Vector{T},t_star,m) where T
    # Theoretical version for univariate gamma convolutions.
    s = θ ./ (1 .- t_star .* θ)
    return [sum(α .* log.(1 .-s)), [sum(α.*(s.^i)) for i in 1:m]...]
end
ugc_k_stats(D::ThorinDistributions.UnivariateGammaConvolution,t_star,m) = ugc_k_stats(D.α,D.θ,t_star,m)

# weird idea, but might work...
function correct_k_stats(k)
    lk = log.(k[2:end])
    ii = findfirst(lk[2:end] .> lk[1:end-1])
    a,b = [ones(ii) (1:ii)] \ lk[1:ii]
    lk[(ii+1):end] = a .+ b .* collect((ii+1):length(lk))
    return [k[1],exp.(lk)...]
end
