"""
    MultivariateGamma(α,θ)

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
struct MultivariateGamma{T<:Real, V<:AbstractVector{T}, M<:Distributions.Gamma{T}} <: Distributions.ContinuousMultivariateDistribution where T
    θ::V
    G::M
end

#### Constructor :
function MultivariateGamma(α::T1,θ::AbstractVector{T2}) where {T1 <: Real, T2<:Real}
     T = Base.promote_eltype([α],θ,[1.0])
     θ = T.(θ)
     MultivariateGamma(α,θ,Distributions.Gamma(T(α),T(1)))
end
MultivariateGamma(α::T1,θ::T2) where {T1<:Real, T2<:Real} = Distributions.Gamma(promote(α,θ)...)

Base.eltype(d::MultivariateGamma) = typeof(d.α)
Base.length(d::MultivariateGamma) = length(d.θ)
function Distributions.insupport(d::MultivariateGamma,x::AbstractVector{T}) where {T <: Real}
    # for theta that are 0, x must be 0
    # for other theta, x/theta must be a constant.
    if any(x[theta .== 0] != 0)
        return false
    end
    return all(y -> isapprox(y,first(x),atol=2*eps(T),rtol=T(0)), (x/theta)[theta .!= 0])
end

#### Sampling
struct MGSPL <: Distributions.Sampleable{Distributions.Multivariate,Distributions.Continuous}
    θ
    G
end
Distributions.sampler(d::MultivariateGamma) = MGSPL(d.θ,d.G)
Base.length(s::MGSPL) = length(s.θ)
function Distributions._rand!(rng::Distributions.AbstractRNG, s::MGSPL, x::AbstractVector{T}) where T<:Real
    x .= s.θ .* rand(rng,s.G)
end

#### Logpdf
function Distributions._logpdf(d::MultivariateGamma{T}, x::AbstractArray) where T<:Real
    if !insupport(d,x)
        return T(-Inf)
    end
    for i in 1:length(x)
        if d.θ[i] != 0
            return log(Distributions.pdf(d.G,x[i]/d.θ[i])/d.θ[i])
        end
    end
    print("we go there")
    return all(x .== T(0)) ? T(0) : T(-Inf) # if all theta are zero, then the distribution is a diract in (0,...,0)
end


mean(d::MultivariateGamma) = d.G.α .* d.θ
var(d::MultivariateGamma) = d.G.α .* d.θ .* d.θ
Statistics.cov(d::MultivariateGamma) = d.G.α .* d.θ[na,:] .* d.θ[:,na]