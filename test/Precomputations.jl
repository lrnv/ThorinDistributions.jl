import ThorinDistributions as TD
using Test

@testset "Precomputations.jl tests" begin
    @testset "precomputations are converted correctly to floats." begin 
        P = TD.get_precomp(Float64,10)
        P2 = TD.get_precomp(BigFloat,20)
        @test all(P.LAGUERRE[1:10,1:10] .== Float64.(P2.LAGUERRE[1:10,1:10]))
    end
end
