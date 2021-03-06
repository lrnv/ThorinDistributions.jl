
mutable struct MoschopoulosParameters{T}
        θ₁::T
        C::T
        to_power::AbstractVector{T}
        γ::AbstractVector{T}
        ρ::T
        δ::AbstractVector{T}
end
function MoschopoulosParameters(α,θ)
    T = Base.promote_eltype(α,θ,[1.0])
    α = T.(α)
    θ = T.(θ)
    δ = T.([1])
    θ₁ = Base.minimum(θ)
    C = exp(sum(α .* log.(θ₁ ./ θ)))
    ρ = sum(α)
    to_power = (-θ₁ ./ θ) .+1
    γ = [sum(α .* to_power)] # gamma1
    return MoschopoulosParameters(θ₁,C,to_power,γ,ρ,δ)
end
Base.eltype(::MoschopoulosParameters{T}) where T = T



"""
    UnivariateGammaConvolution(α,θ)

Constructs a distribution that corresponds to the convolutions of Gamma(α[i],θ[i]) distributions.
The distribution can then be used through several methods,
following the Distributions.jl standard, to obtain pdf, cdf, random samples...

The pdf and cdf are handled by the Moshopoulos algorithm, and random samples by simply adding random gammas.
The code is type stable and handles any <:Real types, given by the parameters.

To fit the distribution, a loglikelyhood approach could be used. A more involved approach from Furman might be coded sometimes (but requires tanh-sinh integration and bigfloats...)


# Examples
```julia-repl
julia> dist = UnivariateGammaConvolution([1,0.5, 3.7],[4,2, 10]);
julia> sample = zeros(Float64,10);
julia> Random.rand!(dist,sample);
julia> pdf.((dist,),sample);
```
"""
struct UnivariateGammaConvolution{T<:Real} <: Distributions.ContinuousUnivariateDistribution where T
    α::AbstractVector{T}
    θ::AbstractVector{T}
    P::MoschopoulosParameters{T}
end

# Constructors :
UnivariateGammaConvolution(α::Real,θ::Real) = Distributions.Gamma(α,θ)
function UnivariateGammaConvolution(α::AbstractVector{T1},θ::AbstractVector{T2}) where {T1 <: Real, T2 <: Real}

    # Start by promoving alpha and theta to the same type :
    T = Base.promote_eltype(α,θ,[1.0]) # At least float.
    α = T.(α)
    θ = T.(θ)

    # Remove alphas and theta that are non-positives
    if any(α .== T(0))
        are_pos = α .* θ .!= T(0)
        α = α[are_pos]
        θ = θ[are_pos]
    end

    # regroup theta that are the sames :
    n = length(θ)
    for i in 1:(n-1)
        if i >= length(θ)
            break
        end
        for j in length(θ):-1:(i+1)
            if θ[i] == θ[j]
                α[i] += α[j]
                deleteat!(α,j)
                deleteat!(θ,j)
                j = j-1
            end
        end
    end

    if length(α) == 1
        return Distributions.Gamma(α[1],θ[1])
    end
    P = MoschopoulosParameters(α,θ)
    α = T.(α)
    θ = T.(θ)
    return UnivariateGammaConvolution(α,θ,MoschopoulosParameters(α,θ))
end

#### Support
Distributions.@distr_support UnivariateGammaConvolution 0.0 Inf

#### Conversions
Base.eltype(d::UnivariateGammaConvolution) = typeof(d.α[1])
Base.convert(::Type{UnivariateGammaConvolution{T}}, d::UnivariateGammaConvolution{S}) where {T <: Real, S <: Real} = UnivariateGammaConvolution(T.(d.α), T.(d.θ))

#### parameters
shapes(d::UnivariateGammaConvolution) = d.α
scales(d::UnivariateGammaConvolution) = d.θ
rates(d::UnivariateGammaConvolution) = 1 / d.θ


#### Statistics
mean(d::UnivariateGammaConvolution) = sum(d.α .* d.θ)
var(d::UnivariateGammaConvolution) = sum(d.α .* d.θ .^ 2)


#### Sampling
function Base.rand(rng::Distributions.AbstractRNG,d::UnivariateGammaConvolution)
    sum(rand.(rng,Distributions.Gamma.(d.α,d.θ)))
end

#### Characteristic functions: pdf, cdf, mgf, cf

# Moshopoulos algorithm for pdf and cdf.
function MoschopoulosAlgo!(d::UnivariateGammaConvolution,x::Real, which)

    @assert(which in ["pdf","cdf"], "which should be etiher pdf or cdf")
    T = Base.eltype(d)
    atol = eps(T)
    rtol = T(0)
    entry_type = typeof(x)
    x = T(x)
    if x < T(0)
        return T(0)
    end
    k = 0
    out = T(0)
    while true
        if length(d.P.δ) < k+1
            # then compute the new deltas:
            n = length(d.P.δ)
            for k in n:(k+1)
                pushfirst!(d.P.γ, sum(d.α .* d.P.to_power .^ (k+1)))
                push!(d.P.δ,sum(d.P.γ[2:end] .* d.P.δ)/(k))
            end
        end
        dist = Distributions.Gamma(T(d.P.ρ + k),T(d.P.θ₁))
        if which == "pdf"
            step = d.P.δ[k+1] * Distributions.pdf(dist,x)
        elseif which == "cdf"
            step = d.P.δ[k+1] * Distributions.cdf(dist,x)
        end
        # if ((!isfinite(step)) & (x > T(0)))
        #     print("x = ", x)
        #     print(d)
        #     error("mince")
        # end
        @assert(!((!isfinite(step)) & (x > T(0))),"inf or nan append, the algorithm did not converge for x = $x")
        out += step
        if isapprox(step,T(0),atol=atol,rtol=rtol)
            break
        end
        k = k+1
    end
    out *= d.P.C
    return entry_type(out)
end

Distributions.pdf(d::UnivariateGammaConvolution,x::Real) = MoschopoulosAlgo!(d,x,"pdf")
Distributions.cdf(d::UnivariateGammaConvolution, x::Real) = MoschopoulosAlgo!(d,x,"cdf")
Distributions.logpdf(d::UnivariateGammaConvolution, x::Real) = log(pdf(d,x))
Distributions.mgf(d::UnivariateGammaConvolution, t::Real) = prod((1 - t .* d.θ) .^ (-d.α))
Distributions.cf(d::UnivariateGammaConvolution, t::Real) = prod((1 - (im * t) .* d.θ) .^ (-d.α))




#
# # Let's now test that :
# dist = UnivariateGammaConvolution([1,0.5, 3.7],[4,2, 10])
# sample = zeros(Float64,10)
# import Random
# Random.rand!(dist,sample)
# x = pdf.((dist,),sample)
#
# # R equivalent :
# # coga::dcoga(1:10,c(0.5,0.5),1/c(2,2))
# #  [1] 0.303265330 0.183939721 0.111565080 0.067667642 0.041042499
# #  [6] 0.024893534 0.015098692 0.009157819 0.005554498 0.003368973
#
# # coga::dcoga(1:10,c(1,0.5),1/c(4,2))
# #  [1] 0.14331842 0.14639660 0.13015318 0.10960590 0.08976271
# #  [6] 0.07231982 0.05766797 0.04567114 0.03600119 0.02828581
# # >
# # perfect !!
#
# #
# # A simple bigfloat test :
#
# sample = zeros(BigFloat,1000)
# import Random
# Random.rand!(dist,sample)
