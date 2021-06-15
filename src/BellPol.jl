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


# We might need a function that trasform moments into cumulants. 
function cumulants_from_moments(μ, mu0 = 1)
    # we ned to construct kappa that are the derivatives of the log times mu 
    n = length(μ)
    M = Bell_partial_matrix(n-1,n-1,μ)
    log_der_fact = [log(mu0),[factorial(i-1) * (-1)^(i-1) * mu0^(-i) for i in 1:(n-1)]...]
    return M''log_der_fact
end

# According to wikipedia, we could also obtain them by recursion: 
function cum_from_mom_rec(μ,mu0=1)
    κ = Array{Any}(undef, length(μ))
    κ[1] = log(mu0)
    κ[2] = μ[1] ./ mu0
    for n in 2:(length(μ)-1)
        κ[n+1] = (μ[n] - sum(binomial(n-1,m-1)*κ[m+1]*μ[n-m] for m in 1:(n-1))) / mu0
    end
    return κ
end
