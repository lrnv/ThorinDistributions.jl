module ThorinDistributions

import Distributions
import Random
import Statistics
import DoubleExponentialFormulas

const na = [CartesianIndex()]
include("Precomputations.jl")
include("UnivariateGammaConvolution.jl")
include("MultivariateGamma.jl")
include("MultivariateGammaConvolution.jl")
include("ConvolutionsAndProducts.jl")
include("LaguerreExpensions.jl")
include("MFKProjection.jl")

end
