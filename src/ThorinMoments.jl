function ηs_from_data!(ηs,D,t_star)
    ηs[:,1] = exp.(t_star .* D)
    m = size(ηs,2)
    for i=1:(m-1)
        ηs[:,i+1] = ηs[:,i] .* D ./ i
    end
end
function η_from_data!(η,D,t_star)

    slack = exp.(t_star .* D)
    n = length(slack)
    η[1] = sum(slack)/n
    m = length(η)
    for i=1:(m-1)
        slack = slack .* D ./ i
        η[i+1] = sum(slack)/n
    end
    return η
end
function τ_from_η!(τ,η)
    η[2:end] ./= η[1]
    τ[1] = log(η[1])
    τ[2] = η[2]
    for k in 3:length(η)
        τ[k] = (k-1)*η[k]
        for j in 1:(k-2)
            τ[k] -= τ[j+1] * η[k-j]
        end
    end
    return τ
end

"""
    thorin_moments(D,t_star,m)

D should be a vector of samples from a distribution, t_star is usually taking to be -1, and m is an integer.

This compute the bare thorin moments, defined as a rescaling of the sample biaised `t_star`-shifted cumulants.
"""
function thorin_moments(D::Vector{T},t_star,m) where T
    τ = zeros(T,m+1)
    η = zeros(T,m+1)
    @views η_from_data!(η,D,t_star)
    τ_from_η!(τ,η)
    return τ
end
@inline κ_from_τ(τ,P::PreComp{T}) where T = τ .* [T(1),P.FACTS[1:(length(τ)-1)]...]
function μ_from_κ(κ,P::PreComp{T}) where T
    μ = zero(κ)
    μ[1] = exp(κ[1])
    for k in 2:length(κ)
        for j in 1:k-1
            μ[k] += μ[j] * κ[k-j+1] * P.BINS[j, k-1] #binomial(big(k-2),big(j-1))
        end
    end
    return μ
end
@inline a_from_μ(μ,P::PreComp{T}) where T = sqrt(2) .* (P.LAGUERRE[1:length(μ),1:length(μ)]'μ)
function empirical_coefs(D::Vector{T},m::M) where {M<:Int, T}
    
    P = get_precomp(T,m)::PreComp{T}
    η = zeros(T,m)
    τ = zeros(T,m)
    μ = zeros(T,m)

    # initialize the computations (k=1)
    slack = exp.(-D)
    η[1] = sum(slack)
    μ[1]= η[1]/length(D)

    # Loop (k > 1)
    @inbounds for k in 2:m
        slack .*= D ./ (k-1)
        η[k] = sum(slack)/η[1]
        τ[k] = (k-1)*η[k]
        @inbounds for j in 2:k-1
            τ[k] -= τ[j]*η[k-j+1]
            μ[k] += τ[j]*μ[k-j+1]
        end
        μ[k] += μ[1]*τ[k]
        μ[k] /= (k-1)
    end
    return P.LAGUERRE2[1:m,1:m]'μ
end

function thorin_moments_old(D::Vector{T},t_star,m) where T
    n = length(D)
    τ = zeros(T,m+1)
    η = zeros(T,m+1)
    ηs = zeros(T,(n,m+1))
    @views ηs_from_data!(ηs,D,t_star)
    η = dropdims(Statistics.mean(ηs,dims=1),dims=1)
    τ_from_η!(τ,η)
    return τ
end

"""
resemps_thorin_moments(M,D,t_star,m)

`M` is an integer
`D` is the data, as a (d,n)-shaped matrix, where d is the numebr of dimensions and n the number of samples.
`t_star` is usually taking to be -1
`m` is an integer.

This resamples the bare thorin moments M times, more efficienty than a simple bootstrap of the `thorin_moments(D,t_star,m)` function.
"""
function resemps_thorin_moments(M,D::AbstractMatrix{T},t_star,m) where {T}
    d,n = size(D)
    ηs = zeros(T,(d,n,m+1))
    τ = zeros(T,(M,d,m+1))
    η = zeros(T,(M,d,m+1))
    for dim in 1:d
        @views ηs_from_data!(ηs[dim,:,:],D[dim,:],t_star)
    end

    # Main loop: 
    for i in 1:M
        println(i)
        η[i,:,:] = dropdims(Statistics.mean(ηs[:,StatsBase.sample(1:n,n,replace=true),:],dims=2),dims=2)
        for dim in 1:d
           @views τ_from_η!(τ[i,dim,:],η[i,dim,:])
        end
    end
    return τ
end
function resemps_thorin_moments(M,D::Vector{T},t_star,m) where T
    return resemps_thorin_moments(M,reshape(D,(1,length(D))),t_star,m)[:,1,:]
end