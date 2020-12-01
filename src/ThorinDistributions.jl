module ThorinDistributions

import Distributions
import Random
import Statistics
using DoubleFloats

setprecision(256)
const ArbT = BigFloat

const na = [CartesianIndex()]

include("UnivariateGammaConvolution.jl")
include("MultivariateGamma.jl")
include("MultivariateGammaConvolution.jl")
include("ConvolutionsAndProducts.jl")
include("LaguerreExpensions.jl")

end
