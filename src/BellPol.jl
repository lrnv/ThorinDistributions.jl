function Bell_partial_matrix(N,K,x)
    M = zeros(eltype(x .+ 1.0),(N+1,K+1))
    for n in 0:N
        for k in 0:n
            if n == 0 && k == 0
                M[n+1,k+1] = 1
            elseif n == 0 || k == 0
                M[n+1,k+1] = 0
            else
                M[n+1,k+1] = sum([binomial(n-1,i-1)*x[i]*M[n-i+1,k] for i in 1:n-k+1])
            end
        end
    end
    return M
end
function cumulants_from_moments(μ, mu0 = 1)
    # we ned to construct kappa that are the derivatives of the log times mu 
    n = length(μ)
    M = Bell_partial_matrix(n-1,n-1,μ)
    log_der_fact = [log(mu0),[factorial(big(i-1)) * (-1)^(i-1) * mu0^(-i) for i in 1:(n-1)]...]
    return M''log_der_fact
end

# According to wikipedia, we could also obtain them by recursion: 
function cum_from_mom_rec!(κ,μ,mu0=1)
    # much faster algorithm ! 
    κ[1] = log(mu0)
    κ[2] = μ[1] ./ mu0
    for n in 2:(length(μ)-1)
        κ[n+1] = (μ[n] - sum(binomial(big(n-1),big(m-1))*κ[m+1]*μ[n-m] for m in 1:(n-1))) / mu0
    end
    return κ
end
function cum_from_mom_rec(μ,mu0=1)
    κ = Array{Any}(undef, length(μ)) # needs to be Array{Any} so that the test with polynomials passes...
    cum_from_mom_rec!(κ,μ,mu0)
    return κ
end
function cum_from_mom_rec!(κ,μ)
    return cum_from_mom_rec!(κ,μ[2:end],μ[1])
end

function thorin_moments(D,t_star,m)
    n = length(D)
    mu = sum(D .^(0:(m+1))' .* exp.(t_star .* D),dims=1)/n
    return [log(mu[1]),(cum_from_mom_rec(mu[2:end],mu[1])[2:end] ./ factorial.(big.(0:(m-1))))...]
end
function resemps_thorin_moments(M,D,t_star,m)
    n = length(D)
    T = eltype(D)
    mu_mat = D .^(0:(m+1))' .* exp.(t_star .* D)
    kappa = zeros(T,(M,m+1))
    mu = zeros(T,(M,m+2))
    facts = [big"1.0",factorial.(big.(0:(m-1)))...]

    Threads.@threads for i in 1:M
        println(i)
        mu[i,:] = StatsBase.mean(mu_mat[StatsBase.sample(1:n,n,replace=true),:],dims=1)
        kappa[i,:] = cum_from_mom_rec!(kappa[i,:], mu[i,2:end],mu[i,1]) ./ facts
    end
    
    return kappa
end