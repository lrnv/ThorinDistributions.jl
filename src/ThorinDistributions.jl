module ThorinDistributions

import Distributions
import Random
import Statistics
import DoubleExponentialFormulas
import LinearAlgebra
import PolynomialRoots

const na = [CartesianIndex()]
include("Precomputations.jl")
include("UnivariateGammaConvolution.jl")
include("MultivariateGamma.jl")
include("MultivariateGammaConvolution.jl")
include("ConvolutionsAndProducts.jl")
include("LaguerreExpensions.jl")
include("MFKProjection.jl")

end
