module ThorinDistributions

import Distributions
import Random
import Statistics
import LinearAlgebra
import PolynomialRoots
import Combinatorics
import DynamicPolynomials
import StaticPolynomials 
import Polynomials
import StatsBase
import SparseArrays
import StaticArrays

const na = [CartesianIndex()]
include("Precomputations.jl")
include("UnivariateGammaConvolution.jl")
include("MultivariateGammaConvolution.jl")
include("ConvolutionsAndProducts.jl")
include("LaguerreExpensions.jl")
include("MFKProjection.jl")
include("kStats.jl")
include("ThorinMoments.jl")
include("HankelMatrices.jl")
include("Dykstra.jl")

end
