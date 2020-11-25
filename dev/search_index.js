var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = ThorinDistributions","category":"page"},{"location":"#ThorinDistributions","page":"Home","title":"ThorinDistributions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A gamma distribution is a distribution with pdf","category":"page"},{"location":"","page":"Home","title":"Home","text":"f(x) = fracx^α-1e^-xθΓ(α)θ^α","category":"page"},{"location":"","page":"Home","title":"Home","text":"As Bondesson shows, based on Thorin work, the class of (weak limit of) independent convolutions of gamma distributions is quite large, closed with respect to independent addition and multiplication of random variables, and contains many interesting distributions.","category":"page"},{"location":"","page":"Home","title":"Home","text":"We implement here a multivariate extensions of these results, and statistical estimation routines to allow for estimation of these distributions through a Laguerre expensions of their densities. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [ThorinDistributions]","category":"page"},{"location":"#ThorinDistributions.MultivariateGamma","page":"Home","title":"ThorinDistributions.MultivariateGamma","text":"MultivariateGamma(α,θ)\n\nConstruct a distribution that correspond to several comonotonous gammas with common shapes α and respectives scales θ[i]. The distribution can then be used through several methods, following the Distributions.jl standard, mainly to obtain samples.\n\nThe code is type stable and handles any <:Real types, given by the parameters.\n\nNote that the supprot of this distribution is a half-line in the d-dimensional space, [0,+∞], with angle given by θ.\n\nExamples\n\njulia> dist = MultivariateGamma(2,[3,4,5]);\njulia> sample = zeros(Float64,(3,10));\njulia> Random.rand!(dist,sample);\n\n\n\n\n\n","category":"type"},{"location":"#ThorinDistributions.MultivariateGammaConvolution","page":"Home","title":"ThorinDistributions.MultivariateGammaConvolution","text":"MultivariateGammaConvolution(α,θ)\n\nConstructs a distribution that corresponds to the convolutions of MutlivariateGamma(α[i],θ[i,:]) distributions. The distribution can then be used through several methods, following the Distributions.jl standard\n\nThe pdf and cdf and not yet coded. The code is type stable and handles any <:Real types, given by the parameters.\n\nFitting th edistribution is not yet possible.\n\nExamples\n\njulia> dist = MultivariateGammaConvolution([2,3,1],[3 0;4 0;0 1]);\njulia> sample = zeros(Float64,(2,10));\njulia> Random.rand!(dist,sample);\n\n\n\n\n\n\n","category":"type"},{"location":"#ThorinDistributions.PreComp-Tuple{Any}","page":"Home","title":"ThorinDistributions.PreComp","text":"PreComp(m)\n\nGiven a precison of computations and a tuple m which gives the size of the future laguerre basis, this fonctions precomputes certain quatities these quatities might be needed later...\n\n\n\n\n\n","category":"method"},{"location":"#ThorinDistributions.UnivariateGammaConvolution","page":"Home","title":"ThorinDistributions.UnivariateGammaConvolution","text":"UnivariateGammaConvolution(α,θ)\n\nConstructs a distribution that corresponds to the convolutions of Gamma(α[i],θ[i]) distributions. The distribution can then be used through several methods, following the Distributions.jl standard, to obtain pdf, cdf, random samples...\n\nThe pdf and cdf are handled by the Moshopoulos algorithm, and random samples by simply adding random gammas. The code is type stable and handles any <:Real types, given by the parameters.\n\nTo fit the distribution, a loglikelyhood approach could be used. A more involved approach from Furman might be coded sometimes (but requires tanh-sinh integration and bigfloats...)\n\nExamples\n\njulia> dist = UnivariateGammaConvolution([1,0.5, 3.7],[4,2, 10]);\njulia> sample = zeros(Float64,10);\njulia> Random.rand!(dist,sample);\njulia> pdf.((dist,),sample);\n\n\n\n\n\n","category":"type"},{"location":"#Distributions.convolve-Tuple{Distributions.Gamma,Distributions.Gamma}","page":"Home","title":"Distributions.convolve","text":"Distributions.convolve(d1,d2)\n\nImplements the special case of covolutions of two Gamma distributions to output UnivariateGammaDistributions. Also works with d1 a gamma and d2 a UnivariateGammaConvolution, and vice versa.\n\n\n\n\n\n","category":"method"},{"location":"#Distributions.product_distribution-Tuple{AbstractArray{var\"#s49\",1} where var\"#s49\"<:ThorinDistributions.UnivariateGammaConvolution}","page":"Home","title":"Distributions.product_distribution","text":"Distributions.product_distribution(d1,d2)\n\nImplements the special case of the product of two gamma or univariate gamma convolutions distributions, and output as the result a multivariate gamma convolution with independent margins.\n\n\n\n\n\n","category":"method"},{"location":"#ThorinDistributions.L2Objective-Tuple{Any,Any}","page":"Home","title":"ThorinDistributions.L2Objective","text":"L2Objective(par,emp_coefs)\n\nA L2 distance to be minimized between laguerre coefficients of MultivariateGammaConvolutions and empirical laguerre coefficients. This distance is some kind of very high degree polynomial * exponentials, so minimizing it is very hard.\n\n\n\n\n\n","category":"method"},{"location":"#ThorinDistributions.L2ObjectiveWithPenalty-Tuple{Any,Any}","page":"Home","title":"ThorinDistributions.L2ObjectiveWithPenalty","text":"L2ObjectiveWithPenalty(par,emp_coefs)\n\nA L2 distance to be minimized between laguerre coefficients of MultivariateGammaConvolutions and empirical laguerre coefficients. This distance is some kind of very high degree polynomial * exponentials, so minimizing it is very hard. This version includes a penalty to force parameters to go towards 0. but it is not yet working correctly.\n\n\n\n\n\n","category":"method"},{"location":"#ThorinDistributions.empirical_coefs-Tuple{Any,Any}","page":"Home","title":"ThorinDistributions.empirical_coefs","text":"empirical_coefs(x,maxp)\n\nCompute the empirical laguerre coefficients of the density of the random vector x, given as a Matrix with n row (number of samples) and d columns (number of dimensions)\n\n\n\n\n\n","category":"method"},{"location":"#ThorinDistributions.get_coefficients-Tuple{Any,Any,Any}","page":"Home","title":"ThorinDistributions.get_coefficients","text":"get_coefficients(α,θ,m)\n\nα should be a vector of shapes of length n, θ should be a matrix of θ of size (n,d) for a MultivariateGammaConvolution with d marginals.\n\nThis function produce the coefficients of the multivariate (tensorised) expensions of the density. It is type stable and quite optimized. it is based on some precomputations that are done in a certain precision.\n\n\n\n\n\n","category":"method"},{"location":"#ThorinDistributions.laguerre_L_2x-Tuple{Any,Any}","page":"Home","title":"ThorinDistributions.laguerre_L_2x","text":"laguerre_L_2x(x,p)\n\nCompute univariate laguerre polynomials Lₚ(2x).\n\n\n\n\n\n","category":"method"},{"location":"#ThorinDistributions.laguerre_density-Tuple{Any,Any}","page":"Home","title":"ThorinDistributions.laguerre_density","text":"laguerre_density(x,coefs)\n\nGiven some laguerre coefficients, computes the correpsonding function at the point x.\n\n\n\n\n\n","category":"method"},{"location":"#ThorinDistributions.laguerre_phi-Tuple{Any,Any}","page":"Home","title":"ThorinDistributions.laguerre_phi","text":"laguerre_phi(x,p)\n\nComputes the function ϕₚ(x) = √2 e⁻ˣ Lₚ(2x). This functions form an orthonormal basis of R₊\n\n\n\n\n\n","category":"method"},{"location":"#ThorinDistributions.laguerre_phi_several_pts-Tuple{Any,Any}","page":"Home","title":"ThorinDistributions.laguerre_phi_several_pts","text":"laguerre_phi_several_pts(x,max_p)\n\nComputes the laguerrephi for each row of x and each p in CartesianIndex(maxp) This is a lot more efficient than broadcasting the laguerrephi function, but this is ugly in the sense that we re-wrote some of the code.\n\n\n\n\n\n","category":"method"},{"location":"#ThorinDistributions.minimum_m-Tuple{Any,Any}","page":"Home","title":"ThorinDistributions.minimum_m","text":"minimum_m(n,d)\n\nComputes the minimum integer m such that m^d > (d+1)n\n\n\n\n\n\n","category":"method"}]
}