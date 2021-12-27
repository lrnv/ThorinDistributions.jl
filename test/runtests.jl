using Test
using TestSetExtensions

@testset ExtendedTestSet "ThorinDistributions tests" begin
    @includetests ARGS
end

# You can use these with Pkg.test("MFKComputation") to run only the tests from the test/MFKComputation.jl file.



# @testset verbose=true "ThorinDistributions.jl" begin
#     include("kStats.jl")
#     include("ThorinMoments.jl")
#     include("UnivariateGammaConvolution.jl")
#     include("HankelMatrices.jl")
# end

