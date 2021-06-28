using Distributions, Random, Statistics, Test
import ThorinDistributions as TD
import DynamicPolynomials, Statistics, StatsBase

####################################
# Previous versions of the code, used now for testing purposes only. 
####################################
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
# According to wikipedia, we could also obtain them by recursion.
# see end of paragraph https://en.wikipedia.org/wiki/Cumulant#Cumulants_and_moments
function cum_from_mom_rec!(κ,μ,mu0=nothing)
    if isnothing(mu0)
        mu0 = μ[1]
        μ = μ[2:end]
    end
    # much faster algorithm ! 
    κ[1] = log(mu0)
    κ[2] = μ[1] ./ mu0
    for n in 2:(length(μ)-1)
        κ[n+1] = (μ[n] - sum(binomial(big(n-1),big(m-1))*κ[m+1]*μ[n-m] for m in 1:(n-1))) / mu0
    end
    return κ
end
function cum_from_mom_rec(μ;mu0)
    if isnothing(mu0)
        mu0 = μ[1]
        μ = μ[2:end]
    end
    κ = Array{Any}(undef, length(μ)) # needs to be Array{Any} so that the test with polynomials passes...
    cum_from_mom_rec!(κ,μ,mu0)
    return κ
end
function old_thorin_moments(D,t_star,m)
    n = length(D)
    mu = sum(D .^(0:(m+1))' .* exp.(t_star .* D),dims=1)/n
    return [log(mu[1]),(cum_from_mom_rec(mu[2:end],mu0=mu[1])[2:end] ./ factorial.(big.(0:(m-1))))...]
end
function old_resemps_thorin_moments(M,D,t_star,m)
    n = length(D)
    T = eltype(D)
    mu_mat = D .^(0:(m+1))' .* exp.(t_star .* D)
    kappa = zeros(T,(M,m+1))
    facts = [T(1),factorial.(T.(0:(m-1)))...]
    for i in 1:M
        @views kappa[i,:] = cum_from_mom_rec!(kappa[i,:], StatsBase.mean(mu_mat[StatsBase.sample(1:n,n,replace=true),:],dims=1)) ./ facts
    end
    return kappa
end
function cum_from_mom_rec_simplified(μ;mu0)
    # for test only
    if isnothing(mu0)
        mu0 = μ[1]
        μ = μ[2:end]
    end
    η = Array{Any}(undef, length(μ)+1)
    τ = Array{Any}(undef, length(μ)+1)
    η .= [mu0,(μ ./ factorial.(big.(1:length(μ))))...]
    η[2:end] ./= mu0
    τ[1] = log.(mu0)
    τ[2] = η[2]
    for k in 3:length(η)
        τ[k] = (k-1)*η[k]
        for j in 1:(k-2)
            τ[k] -= τ[j+1] * η[k-j]
        end
    end
    κ = τ .* [1,factorial.(big.(0:(length(τ)-2)))...]
    return κ[1:end-1]
end 

@testset "BellPol.jl" begin

    DynamicPolynomials.@polyvar μ[1:10];
    κ1 = cumulants_from_moments(μ,1);
    κ2 = cum_from_mom_rec(μ,mu0=1);
    κ3 = cum_from_mom_rec_simplified(μ,mu0=1);
    κ4 = cumulants_from_moments(μ,0.5);
    κ5 = cum_from_mom_rec(μ,mu0=0.5);
    κ6 = cum_from_mom_rec_simplified(μ,mu0=0.5);

    @testset "firsts cumulants corretness, mu0=1" begin
        @test κ1[1] == 0
        @test κ1[2] == μ[1]
        @test κ1[3] == μ[2] - μ[1]^2
        @test κ1[4] == 2.0μ[1]^3 - 3.0μ[1]μ[2] + μ[3]
        @test κ1[5] ==  -6.0μ[1]^4 + 12.0μ[1]^2*μ[2] - 4.0μ[1]μ[3] - 3.0μ[2]^2 + μ[4]
        @test κ1[6] ==   24.0μ[1]^5 - 60.0μ[1]^3*μ[2] + 20.0μ[1]^2*μ[3] + 30.0μ[1]*μ[2]^2 - 5.0μ[1]*μ[4] - 10.0μ[2]*μ[3] + μ[5]
    end

    @testset "cumulants recursivity" begin
        @test κ1 == κ2
        @test κ4 == κ5
    end

    @testset "η/τ simplification" begin
        @test all(κ3 .≈ κ1)
        @test all(κ3 .≈ κ2)
        @test all(κ6 .≈ κ4)
        @test all(κ6 .≈ κ5)
    end

    @testset "Bell partial matrix" begin
        M1 = [1   0      0       0       0       0       0      0     0   0  0
        0   1      0       0       0       0       0      0     0   0  0
        0   2      1       0       0       0       0      0     0   0  0
        0   3      6       1       0       0       0      0     0   0  0
        0   4     24      12       1       0       0      0     0   0  0
        0   5     80      90      20       1       0      0     0   0  0
        0   6    240     540     240      30       1      0     0   0  0
        0   7    672    2835    2240     525      42      1     0   0  0
        0   8   1792   13608   17920    7000    1008     56     1   0  0
        0   9   4608   61236  129024   78750   18144   1764    72   1  0
        0  10  11520  262440  860160  787500  272160  41160  2880  90  1]
        @test Int.(Bell_partial_matrix(10,10,1:30)) == M1
    end

    @testset "Resemp is OK" begin
        D = BigFloat.(rand(LogNormal(),100));
        M = 10
        Random.seed!(123)
        x1 = old_resemps_thorin_moments(M,D,-1,10)
        Random.seed!(123)
        x2 = TD.resemps_thorin_moments(M,D,-1,10)
        @test x1 ≈ x2
        Random.seed!(123)
        x3 = TD.resemps_thorin_moments(M,[D 2D]',-1,10)
        @test x1 ≈ x3[:,1,:]

    end
end
