import ThorinDistributions as TD
import Polynomials
import DynamicPolynomials
import Combinatorics
using Test


@testset "kStats.jl" begin 
    @testset "better_stirling correctness" begin
        @test TD._better_stirling(10,5)[10,5]==Combinatorics.stirlings2(10,5)
    end
    
    @testset "_fd correctness" begin
        @test TD._fd(3,0) == Polynomials.Polynomial([0,-6,11,-6,1])
        @test TD._fd(3,1) == Polynomials.Polynomial([-6,11,-6,1])
        @test TD._fd(3,2) == Polynomials.Polynomial([6,-5,1])
    end
    
    @testset "_compute_k correctness" begin
        DynamicPolynomials.@polyvar N
        DynamicPolynomials.@polyvar S[1:9]
        result = TD._compute_k(N,S);
        Test.@test result[1] == S[1]/N
        Test.@test result[2] == (N * S[2] - S[1]^2)/(N*(N-1))
        Test.@test result[3] == (2S[1]^3 - 3N*S[1]*S[2] +N^2 * S[3]) / (N*(N-1)*(N-2))
        Test.@test result[4] == (-6*S[1]^4 + 12 * N * S[1]^2 * S[2] - 3 * N * (N-1) * S[2]^2 - 4 * N * (N+1) * S[1] * S[3] + N^2*(N+1)*S[4])/(N*(N-1)*(N-2)*(N-3))
    end
end

