import DynamicPolynomials
import ThorinDistributions as TD
using Test

@testset "BellPol.jl" begin
        
    DynamicPolynomials.@polyvar μ[1:20];
    κ = TD.cumulants_from_moments(μ);
    κ2 = TD.cum_from_mom_rec(μ);
    κ3 = TD.cumulants_from_moments(μ,0.5);
    κ4 = TD.cum_from_mom_rec(μ,mu0=0.5);

    @testset "firsts cumulants corretness, mu0=1" begin
        @test κ[1] == 0
        @test κ[2] == μ[1]
        @test κ[3] == μ[2] - μ[1]^2
        @test κ[4] == 2.0μ[1]^3 - 3.0μ[1]μ[2] + μ[3]
        @test κ[5] ==  -6.0μ[1]^4 + 12.0μ[1]^2*μ[2] - 4.0μ[1]μ[3] - 3.0μ[2]^2 + μ[4]
        @test κ[6] ==   24.0μ[1]^5 - 60.0μ[1]^3*μ[2] + 20.0μ[1]^2*μ[3] + 30.0μ[1]*μ[2]^2 - 5.0μ[1]*μ[4] - 10.0μ[2]*μ[3] + μ[5]
    end

    @testset "cumulants recursivity" begin
        @test κ == κ2
        @test κ3 == κ4
    end

    M1 = [1   0      0       0       0       0       0      0     0   0  0
    0   1      0       0       0       0       0      0     0   0  0
    0   2      1       0       0       0       0      0     0   0  0
    0   3      6       1       0       0       0      0     0   0  0
    0   4     24      12       1       0       0      0     0   0  0
    0   5     80      90      20       1       0      0     0   0  0
    0   6    240     540     240      30       1      0     0   0  0
    0   7    672    2835    2240     525      42      1     0   0  0
    0   8   1792   13608   17920    7000    1008     56     1   0  0
    0   9   4608   61236  129024   78750   18144   1764    72   1  0
    0  10  11520  262440  860160  787500  272160  41160  2880  90  1]



    @testset "Bell partial matrix" begin
        @test Int.(TD.Bell_partial_matrix(10,10,1:30)) == M1
    end




    # import DynamicPolynomials
    # DynamicPolynomials.@polyvar x[1:20]
    # M = Bell_partial_matrix(20,20,x)
    # Int.(Bell_partial_matrix(10,10,1:30))
    # sum(M,dims=2)[1:10] # this is OK as wikipedia shows. 



end
