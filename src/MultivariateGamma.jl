using Distributions
import Statistics

na = [CartesianIndex()]

"""
UnivariateGammaConvolution(α,θ)

Construct a distribution that correspond to several comonotonous gammas with common shapes α and respectives scales θ[i].
The distribution can then be used through several methods,
following the Distributions.jl standard, mainly to obtain samples.

The code is type stable and handles any <:Real types, given by the parameters.

Note that the supprot of this distribution is a half-line in the d-dimensional space, [0,+∞], with angle given by θ.

# Examples
```julia-repl
julia> dist = MultivariateGamma(2,[3,4,5]);
julia> sample = zeros(Float64,(3,10));
julia> Random.rand!(dist,sample);
```
"""
struct MultivariateGamma{T<:Real} <: Distributions.ContinuousMultivariateDistribution where T
    α::T
    θ::AbstractVector{T}
    Γα::Distributions.Gamma{T}
end

#### Constructor :
function MultivariateGamma(α::T1,θ::AbstractVector{T2}) where {T1 <: Real, T2<:Real}
     T = Base.promote_eltype([α],θ,[1.0])
     α = T(α)
     θ = T.(θ)
     MultivariateGamma(α,θ,Distributions.Gamma(α,T(1)))
end
MultivariateGamma(α::T1,θ::T2) where {T1<:Real, T2<:Real} = Distributions.Gamma(promote(α,θ)...)

#### eltype, length, support
eltype(d::MultivariateGamma) = typeof(d.α)
Base.length(d::MultivariateGamma) = length(d.θ)
function Distributions.insupport(d::MultivariateGamma,x::AbstractVector{T}) where {T <: Real}
    # for theta that are 0, x must be 0
    # for other theta, x/theta must be a constant.
    j = 0
    for i in 1:length(x)
        if d.θ[i] == T(0)
            if not x[i] == T(0)
                return false
            end
        else
            j = i
        end
    end
    if j == 0
        # then all theta are 0 and all x are 0
        return true
    end
    constant = x[j]/d.θ[j]
    for i in 1:length(x)
        if d.θ[i] != 0
            # for stability issue, we doubled the tolerence of the type. Otherwise even simulations are not inbounds, which causes pdf == 0 on simulated data !
            if !isapprox(x[i]/d.θ[i],constant,atol=2*eps(T),rtol=T(0))
                return false
            end
        end
    end
    return true
end

#### Sampling
struct MGSPL <: Distributions.Sampleable{Multivariate,Continuous}
    α
    θ
    Γα
end
Distributions.sampler(d::MultivariateGamma) = MGSPL(d.α,d.θ,d.Γα)
Base.length(s::MGSPL) = length(s.θ)
function Distributions._rand!(rng::Distributions.AbstractRNG, s::MGSPL, x::AbstractVector{T}) where T<:Real
    x .= s.θ .* rand(rng,s.Γα) # simulate a gamma(alpha,1) and multiply by theta
end

#### Logpdf
function Distributions._logpdf(d::MultivariateGamma{T}, x::AbstractArray) where T<:Real
    if !insupport(d,x)
        return T(-Inf)
    end
    for i in 1:length(x)
        if d.θ[i] != 0
            return log(Distributions.pdf(d.Γα,x[i]/d.θ[i])/d.θ[i])
        end
    end
    print("we go there")
    return all(x .== T(0)) ? T(0) : T(-Inf) # if all theta are zero, then the distribution is a diract in (0,...,0)
end


mean(d::MultivariateGamma) = d.α .* d.θ
var(d::MultivariateGamma) = d.α .* d.θ .* d.θ
Statistics.cov(d::MultivariateGamma) = d.α .* d.θ[na,:] .* d.θ[:,na]


# # Okay this should be enough to obtain a pdf
#
# dist = MultivariateGamma(2,[3,4,5])
# sample = zeros(Float64,(3,10))
# import Random
# Random.rand!(dist,sample)
# display(sample)
#
#
# display(Distributions.pdf(dist,sample))
# display(Distributions.pdf(Distributions.Gamma(dist.α,dist.θ[1]),sample[1,:]))
