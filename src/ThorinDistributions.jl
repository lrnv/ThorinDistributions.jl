module ThorinDistributions

import Distributions
import Random
import Statistics
import DoubleExponentialFormulas
import LinearAlgebra
import PolynomialRoots
import Combinatorics
import DynamicPolynomials
import StaticPolynomials 
import Polynomials
import StatsBase

const na = [CartesianIndex()]
include("Precomputations.jl")
include("UnivariateGammaConvolution.jl")
include("MultivariateGamma.jl")
include("MultivariateGammaConvolution.jl")
include("ConvolutionsAndProducts.jl")
include("LaguerreExpensions.jl")
include("MFKProjection.jl")
include("kStats.jl")
include("BellPol.jl")

end
