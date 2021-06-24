using ThorinDistributions
using Test

@testset verbose=true "ThorinDistributions.jl" begin
    include("kStats_tests.jl")
    include("BellPol_tests.jl")
end

