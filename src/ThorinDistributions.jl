module ThorinDistributions

import Distributions
import Random
import Statistics
#using ArbNumerics

na = [CartesianIndex()]

include("UnivariateGammaConvolution.jl")
include("MultivariateGamma.jl")
include("MultivariateGammaConvolution.jl")
include("ConvolutionsAndProducts.jl")
include("LaguerreExpensions.jl")

end
