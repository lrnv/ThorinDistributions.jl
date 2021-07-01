import ThorinDistributions
using DynamicPolynomials, Random
Random.seed!(123)

function get_hankels_truth(s,a,b)
    m = length(s)-1
    if iseven(m)
        # m = 2n
        n = Int(m//2)
        H_low = [s[i+j-1] for i in 1:n+1,j in 1:n+1]
        H_up = [(a+b)*s[i+j] - s[i+j+1] - a*b*s[i+j-1] for i in 1:n,j in 1:n]
    else
        # m = 2n+1
        n = Int((m-1)//2)
        H_low = [s[i+j] - a*s[i+j-1] for i in 1:n+1,j in 1:n+1]
        H_up = [b*s[i+j-1] - s[i+j] for i in 1:n+1,j in 1:n+1]
    end
    return H_low,H_up
end

@testset "HankelMatrices.jl tests" begin 

    m=20
    @polyvar x[1:m+1]
    H_low, H_up = vec.(ThorinDistributions.get_hankels(x,2,3))
    H_low_op, H_up_op = ThorinDistributions.vec_H.(ThorinDistributions.hankel_operators(m,2,3))
    
    @testset "coherence of hankel_operator" begin
        @test H_low_op*x == H_low
        @test H_up_op*x == H_up
    end

    a = rand()
    b = rand()
    a,b = min(a,b),max(a,b)

    @testset "coherence of get_hankels for random bounds" begin 
        @test get_hankels_truth(x,3,4) == ThorinDistributions.get_hankels(x,3,4)
        @test get_hankels_truth(x,a,b) == ThorinDistributions.get_hankels(x,a,b)
    end

    x = randn(100)

    @testset "coherence for random values" begin
        @test get_hankels_truth(x,0,1) == ThorinDistributions.get_hankels(x,0,1)
        @test get_hankels_truth(x,a,b) == ThorinDistributions.get_hankels(x,a,b)
    end
end