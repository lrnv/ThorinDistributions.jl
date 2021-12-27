# The goal of this file is to provide tools that implements the MFK algorithm from these two references: 
# @article{furman2017,
#   title = {On {{Log}}-{{Normal Convolutions}}: {{An Analytical}}-{{Numerical Method With Applications}} to {{Economic Capital Determination}}},
#   shorttitle = {On {{Log}}-{{Normal Convolutions}}},
#   author = {Furman, Edward and Hackmann, Daniel and Kuznetsov, Alexey},
#   year = {2017},
#   issn = {1556-5068},
#   abstract = {We put forward an efficient algorithm for approximating the sums of independent and lognormally distributed random variables. Namely, by merging tools from probability theory and numerical analysis, we are able to compute the cumulative distribution functions of the just-mentioned sums with any desired precision. Importantly, our algorithm is fast and can tackle equally well sums with just a few or thousands of summands. We illustrate the effectiveness of the new method in the contexts of the individual and collective risk models, aggregate economic capital determination, and economic capital allocation.},
#   journal = {SSRN Electronic Journal},
#   language = {en}
# }

# @article{miles2019,
#   ids = {mile,milesa},
#   title = {Risk Aggregation: {{A}} General Approach via the Class of {{Generalized Gamma Convolutions}}},
#   author = {Miles, Justin and Furman, Edward and Kuznetsov, Alexey},
#   year = {2019},
#   journal = {Variance},
#   keywords = {gamma}
# }

"""

    compute_g(dist,n, integrator)

This function will compute the n first theretical (-1)-exponentially shifted moments of a distribution, 
using takashi-mori tanh-sinh exponentials formulas from the DoubleExponentialFormulas package. 

This might take a long time. As an integrator, you should pass the result of `QuadDE(BigFloat)` after setting enough precision.

"""
function compute_g(dist,n, integrator)
    g = Array{BigFloat}(undef, 1,2n+1)
    residuals_g = deepcopy(g)
    for i in 0:(2n)
        g[i+1],residuals_g[i+1] = integrator(x -> (-x)^(i) * Distributions.pdf(dist,x) * exp(-x), 0, +Inf)
        # print("g_{",i,"} = ",Float64(g[i+1]),", rez = ",Float64(residuals_g[i+1]),"\n")
    end
    return g
end


"""

    E_from_g(g)

Simply computes the empirical laguerre coefficients from the exponentially shifted moments, 
if you computed those moments from a theoretical distribution.

"""
function E_from_g(g)
    # this function should compute laguerre coefficients from g. 
    a = deepcopy(g)
    for i in 1:length(a)
        a[i] = sqrt(2) * sum(binomial(big(i-1),big(k))*big(2)^(k)/factorial(big(k)) * g[k+1] for k in 0:(i-1))
    end
    return a
end


function build_s!(s,g,facts)
    s[1] = -g[2]/g[1]
    for k in 1:(length(s)-1)
        s[k+1] = g[k+2] / facts[k+1]
        for i in 0:(k-1)
            s[k+1] += s[i+1] * g[k-i+1] / facts[k-i+1]
        end
        s[k+1] = - s[k+1]/g[1]
    end
end

function s_from_k(k)
    if isodd(length(k))
        n = Int((length(k)-1)/2)
    else
        n = Int(length(k)/2) - 1
    end
    # n = Int(round(length(k)/2))
    s = k[2:2n+1] .* (-1) .^(2:(2n+1))
    return s
end

function MFK_end(s::Vector{T}) where T
    n = Int(length(s)//2)
    S = zeros(T,(n,n))
    for i in 0:(n-1)
        for j in 0:(n-1)
            S[i+1,j+1] = s[i+j+1]
        end
    end

    sol_b = LinearAlgebra.Symmetric(S) \ (-s[(n+1):end])
    b = deepcopy(sol_b)
    append!(b,1)
    b = reverse(b)
    b_deriv = reverse(sol_b) .* (1:n)

    a = zeros(T,n)
    a[1] = s[1]
    for k in 1:(n-1)
        a[k+1] = s[k+1]
        for i in 0:(k-1)
            a[k+1] = a[k+1] + b[i+2] * s[k-i]
        end
    end

    z = real.(PolynomialRoots.roots(b, polish=true))

    beta = -z .-1
    alpha = deepcopy(beta)

    for i in 1:length(alpha)
        rez_num = 0
        rez_denom = 0
        for k in 1:length(a)
            rez_num += a[k] * z[i]^(k-1)
        end
        for k in 1:length(b_deriv)
            rez_denom += b_deriv[k] * z[i]^(k-1)
        end
        alpha[i] = rez_num/rez_denom
    end
    theta = 1 ./beta
    return alpha,theta
end





"""

    MFK_Projection(g_integrals,n_gammas)

From a set of g_integrals computed by the compute_g function, this function runs the algorithm from Miles, Furman & Kuznetsov 
to produce a univariate gamma convolution. 

This algorithm works only if the distribution you computed the g_integrals from is, indeed, a generalized gamma convolution. 
otherwise, it might fail and return garbage. 

"""
function MFK_Projection(g_integrals::Vector{T},n) where T
    facts = factorial.(T.(0:2n))
    s = Array{T}(undef, 2n)
    build_s!(s,g_integrals,facts)
    alpha,beta = MFK_end(s)
    return ThorinDistributions.UnivariateGammaConvolution(Float64.(alpha), Float64.(1 ./ beta))
end