# The following method was removed due to the fact that it is a type piracy from Distributions.jl
# Too bad; 


#Distributions.convolve(d1::Distributions.Gamma,d2::Distributions.Gamma) = UnivariateGammaConvolution([d1.α,d2.α],[d1.θ,d2.θ])
Distributions.convolve(d1::Distributions.Gamma,d2::UnivariateGammaConvolution) = UnivariateGammaConvolution([d1.α,d2.α...], [d1.θ,d2.θ...])
Distributions.convolve(d1::UnivariateGammaConvolution,d2::Distributions.Gamma) = Distribution.convolve(d2,d1)
# Products of UnivariateGammaConvolution

"""
    Distributions.product_distribution(d1,d2)

Implements the special case of the product of two gamma or univariate gamma convolutions distributions, and
output as the result a multivariate gamma convolution with independent margins.
"""
function Distributions.product_distribution(dists::AbstractVector{<:UnivariateGammaConvolution})

    # number of gammas :
    T = Base.promote_eltype(dists)
    d = length(dists)
    ns = [length(d.α) for d in dists]
    n = sum(ns)
    α = zeros(T,n)
    θ = zeros(T,(n,d))
    current_n = 1
    for i in 1:d
        α[current_n:(current_n-1+ns[i])] = dists[i].α
        θ[current_n:(current_n-1+ns[i]),i] = dists[i].θ
        current_n +=ni
    end
    return MultivariateGammaConvolution(α,θ)
end
