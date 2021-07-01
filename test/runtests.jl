using Test
using TestSetExtensions

@testset ExtendedTestSet "ThorinDistributions tests" begin
    @includetests ARGS
end



# @testset verbose=true "ThorinDistributions.jl" begin
#     include("kStats.jl")
#     include("ThorinMoments.jl")
#     include("UnivariateGammaConvolution.jl")
#     include("HankelMatrices.jl")
# end

