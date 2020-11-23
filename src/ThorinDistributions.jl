module ThorinDistributions

import Distributions
import Random
import Statistics

na = [CartesianIndex()]

include("UnivariateGammaConvolution.jl")
include("MultivariateGamma.jl")
include("MultivariateGammaConvolution.jl")
include("ConvolutionsAndProducts.jl")

end
