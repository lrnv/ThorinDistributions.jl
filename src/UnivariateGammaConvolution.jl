
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

n(x::UnivariateGammaConvolution) = length(x.α)

function Base.show(io::IO, m::UnivariateGammaConvolution) 
    println("Univariate Gamma Convolutions with parametrisation:")
    display([Text.(["α" "θ"]); [m.α m.θ]])
end

# Constructors :
UnivariateGammaConvolution(α::Real,θ::Real) = Distributions.Gamma(α,θ)
function UnivariateGammaConvolution(α::AbstractVector{T1},θ::AbstractVector{T2}) where {T1 <: Real, T2 <: Real}

    # Start by promoving alpha and theta to the same type :
    T = Base.promote_eltype(α,θ,[1.0]) # At least float.
    α = T.(α)
    θ = T.(θ)
    tol = eps(T) # Arbitrary fixed tolerence ! 
    
    order = sortperm(θ)
    n = length(θ)
    θ = θ[order]
    α = α[order]
    for i in 1:n
        if i < length(θ)
            for j in (i+1):n
                if j <= length(θ)
                    if abs(θ[i] - θ[j]) <= tol
                        new_α = α[i]+α[j]
                        new_θ = (α[i]*θ[i]+α[j]*θ[j])/(new_α)
                        α[i] = new_α
                        θ[i] = new_θ
                        deleteat!(α,j)
                        deleteat!(θ,j)
                        j = j-1
                    end
                end
            end
        end
    end
    to_be_kept = (α .> tol) .& (θ .> tol) .* (α.*θ .> tol)
    α = α[to_be_kept]
    θ = θ[to_be_kept]

    if length(α) == 1
        return Distributions.Gamma(α[1],θ[1])
    end
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
