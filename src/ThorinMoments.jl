function ηs_from_data!(ηs,D,t_star)
    ηs[:,1] = exp.(t_star .* D)
    m = size(ηs,2)
    for i=1:(m-1)
        ηs[:,i+1] = ηs[:,i] .* D ./ i
    end
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

M is an integer, D should be a vector of samples from a distribution, t_star is usually taking to be -1, and m is an integer.

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
    Threads.@threads for i in 1:M
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