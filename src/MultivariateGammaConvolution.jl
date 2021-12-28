"""
    MultivariateGammaConvolution(α,θ)

Constructs a distribution that corresponds to the convolutions of MutlivariateGamma(α[i],θ[i,:]) distributions.
The distribution can then be used through several methods, following the Distributions.jl standard

The pdf and cdf and not yet coded.
The code is type stable and handles any <:Real types, given by the parameters.

Fitting th edistribution is not yet possible.

# Examples
```julia-repl
julia> dist = MultivariateGammaConvolution([2,3,1],[3 0;4 0;0 1]);
julia> sample = zeros(Float64,(2,10));
julia> Random.rand!(dist,sample);

```
"""
struct MultivariateGammaConvolution{T<:Real, V<:AbstractVector{T}, M<: AbstractMatrix{T}} <: Distributions.ContinuousMultivariateDistribution where T
    α::V
    θ::M
    constant::Int
end

n(x::MultivariateGammaConvolution) = length(x.α)

function Base.show(io::IO, m::MultivariateGammaConvolution) 
    println("Multivariate Gamma Convolutions with parametrisation:")
    display([m.α m.θ])
end

#### Constructor :
function MultivariateGammaConvolution(α::AbstractVector{T1},θ::AbstractMatrix{T2}) where {T1 <: Real, T2<:Real}

    @assert(length(α) == size(θ,1),"You must provide α with same length as the number of rows in θ")
    # Propote eltypes :
    T = Base.promote_eltype(α,θ,[1.0])
    α = T.(α)
    θ = T.(θ) # the size of theta is (n,d)
    θ[θ .< eps(T)] .= 0
    # Remove alphas that are non-positives or thetas that are all zero.
    n = length(α)
    are_pos = BitVector(undef,n)
    for i in 1:n
        are_pos[i] = α[i] > eps(T)
        are_pos[i] &= any(θ[i,:] .> eps(T))
    end
    α = α[are_pos]
    θ = θ[are_pos,:] 
    θ = θ[sortperm(α),:]
    sort!(α)
    return MultivariateGammaConvolution(α,θ,1)
end
# MultivariateGammaConvolution(α::T1,θ::AbstractVector{T2}) where {T1<:Real, T2<:Real} = MultivariateGamma(promote(α,θ)...)
MultivariateGammaConvolution(α::AbstractVector{T1},θ::AbstractVector{T2}) where {T1<:Real, T2<:Real} = UnivariateGammaConvolution(promote(α,θ)...)


#### eltype, length, support
Base.eltype(d::MultivariateGammaConvolution) = typeof(d.α)
Base.length(d::MultivariateGammaConvolution) = size(d.θ,2)
function Distributions.insupport(d::MultivariateGammaConvolution,x::AbstractVector{T}) where {T <: Real}
    return all(x > T(0))
end

#### Sampling
function Distributions._rand!(rng::Distributions.AbstractRNG, s::MultivariateGammaConvolution, x::AbstractVector{T}) where T<:Real
    x .= s.θ'rand.(rng,Distributions.Gamma.(s.α,1))
end
function Distributions._rand!(rng::Distributions.AbstractRNG, s::MultivariateGammaConvolution, A::DenseMatrix{T}) where T<:Real
    Gs = Distributions.Gamma.(s.α,1)
    n = size(A,2)
    A .= s.θ'transpose(hcat(rand.(rng,Gs,n)...)) 
end
function Base.rand(rng::Distributions.AbstractRNG,d::MultivariateGammaConvolution)
    d.θ'rand.(rng,Distributions.Gamma.(d.α,1))
end

#### Logpdf
function Distributions._logpdf(d::MultivariateGammaConvolution{T}, x::AbstractArray) where T<:Real
    if !insupport(d,x)
        return T(-Inf)
    end
    error("Not implemented yet.")
end


mean(d::MultivariateGammaConvolution) = sum(d.α .* d.θ,axis=1)
var(d::MultivariateGammaConvolution) = sum(d.α .* d.θ .* d.θ,axis=1)