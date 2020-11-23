using Distributions
import Statistics

na = [CartesianIndex()]


"""
UnivariateGammaConvolution(α,θ)

Constructs a distribution that corresponds to the convolutions of MutlivariateGamma(α[i],θ[i,:]) distributions.
The distribution can then be used through several methods, following the Distributions.jl standard

The pdf and cdf and not yet coded.
The code is type stable and handles any <:Real types, given by the parameters.

Fitting th edistribution is not yet possible.

# Examples
```julia-repl
julia> dist = MultivariateGamma(2,[3,4,5]);
julia> sample = zeros(Float64,(3,10));
julia> Random.rand!(dist,sample);
```
"""
struct MultivariateGammaConvolution{T<:Real} <: Distributions.ContinuousMultivariateDistribution where T
    α::AbstractVector{T}
    θ::AbstractMatrix{T}
    constants
end

#### Constructor :
function MultivariateGammaConvolution(α::AbstractVector{T1},θ::AbstractMatrix{T2}) where {T1 <: Real, T2<:Real}


     @assert(length(α) == size(θ,1),"You must provide α with same length as the number of rows in θ")
     # Propote eltypes :
     T = Base.promote_eltype(α,θ,[1.0])
     α = T.(α)
     θ = T.(θ) # the size of theta is (n,d)

     # Remove alphas that are non-positives or thetas that are all zero.
     if any(α .== T(0))
         are_pos = α .!= T(0) | [all(θ[i,:] == 0) for i in 1:size(θ,1)]
         α = α[are_pos]
         θ = θ[are_pos,:]
     end

     # regroup theta that are the sames :
     n = size(θ,1)
     for i in 1:(n-1)
         if i >= size(θ,1)
             break
         end
         for j in (i+1):size(θ,1)
             if all(θ[i,:] == θ[j,:])
                 α[i] += α[j]
                 deleteat!(α,j)
                 θ = θ[[1:(j-1),(j+1):size(θ,1)],:]
                 j = j-1
             end
         end
     end
     if length(α) == 1
         return Distributions.MultivariateGamma(α,θ[:,1])
     end
     return MultivariateGammaConvolution(α,θ, 1)
end
MultivariateGammaConvolution(α::T1,θ::AbstractVector{T2}) where {T1<:Real, T2<:Real} = MultivariateGamma(promote(α,θ)...)
MultivariateGammaConvolution(α::AbstractVector{T1},θ::AbstractVector{T2}) where {T1<:Real, T2<:Real} = UnivariateGammaConvolution(promote(α,θ)...)


#### eltype, length, support
eltype(d::MultivariateGammaConvolution) = typeof(d.α)
Base.length(d::MultivariateGammaConvolution) = size(d.θ,2)
function Distributions.insupport(d::MultivariateGamma,x::AbstractVector{T}) where {T <: Real}
    return all(x > T(0))
end

#### Sampling
struct MGCSPL <: Distributions.Sampleable{Multivariate,Continuous}
    Γs::AbstractVector{MGSPL}
end
Distributions.sampler(d::MultivariateGammaConvolution) = MGCSPL([Distributions.sampler(MultivariateGamma(d.α[i],d.θ[i,:])) for i in 1:length(d.α)])
Base.length(s::MGCSPL) = length(s.Γs[1].θ)
function Distributions._rand!(rng::Distributions.AbstractRNG, s::MGCSPL, x::AbstractVector{T}) where T<:Real
    x .= T(0)
    for i in 1:length(s.Γs)
        x .+= rand(rng,s.Γs[i])
    end
end

#### Logpdf
function Distributions._logpdf(d::MultivariateGamma{T}, x::AbstractArray) where T<:Real
    if !insupport(d,x)
        return T(-Inf)
    end
    error("Not implemented yet.")
end


# mean(d::MultivariateGamma) = d.α .* d.θ
# var(d::MultivariateGamma) = d.α .* d.θ .* d.θ
# Statistics.cov(d::MultivariateGamma) = d.α .* d.θ[na,:] .* d.θ[:,na]


# Okay this should be enough to obtain a pdf

dist = MultivariateGammaConvolution([2,3,1],[3 0;4 0;0 1])
sample = zeros(Float64,(2,10))
import Random
Random.rand!(dist,sample)
display(sample)
#
#
# display(Distributions.pdf(dist,sample))
# display(Distributions.pdf(Distributions.Gamma(dist.α,dist.θ[1]),sample[1,:]))
