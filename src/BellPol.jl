function ηs_from_data!(ηs,D,t_star,m)
    ηs[:,1] = exp.(t_star .* D)
    for i in 1:m
        ηs[:,i+1] = ηs[:,i] .* D ./ i
    end
end
function τ_from_η!(τ,η)
    η[2:end] ./= η[1]
    τ[1] = log.(η[1])
    τ[2] = η[2]
    for k in 3:length(η)
        τ[k] = (k-1)*η[k]
        for j in 1:(k-2)
            τ[k] -= τ[j+1] * η[k-j]
        end
    end
    return τ
end
function thorin_moments(D::Vector{T},t_star,m) where T
    n = length(D)
    τ = zeros(T,m+1)
    η = zeros(T,m+1)
    ηs = zeros(T,(n,m+1))
    @views ηs_from_data!(ηs,D,t_star,m)
    η = Statistics.mean(ηs,dims=1)
    τ_from_η!(τ,η)
    return τ
end
function resemps_thorin_moments(M,D::AbstractMatrix{T},t_star,m) where {T}
    d,n = size(D)
    ηs = zeros(T,(d,n,m+1))
    τ = zeros(T,(M,d,m+1))
    η = zeros(T,(M,d,m+1))
    for dim in 1:d
        @views ηs_from_data!(ηs[dim,:,:],D[dim,:],t_star,m)
    end

    # Main loop: 
    for i in 1:M
        η[i,:,:] = Statistics.mean(ηs[:,StatsBase.sample(1:n,n,replace=true),:],dims=2)
        for dim in 1:d
           @views τ_from_η!(τ[i,dim,:],η[i,dim,:])
        end
    end
    return τ
end
function resemps_thorin_moments(M,D::Vector{T},t_star,m) where T
    return resemps_thorin_moments(M,reshape(D,(1,length(D))),t_star,m)[:,1,:]
end