using Test

@testset verbose=true "ThorinDistributions.jl" begin
    include("kStats_tests.jl")
    include("ThorinMoments_tests.jl")
    include("UnivariateGammaConvolution_tests.jl")
end

