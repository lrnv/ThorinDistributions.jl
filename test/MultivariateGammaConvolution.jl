import ThorinDistributions as TD
using Test
import Random, Distributions

@testset "MultivariateGammaConvolution.jl tests" begin 
    
    # Making data: 
    Random.seed!(12)
    N,n,d = 1000,20,10
    α0 = randn(n).^2;
    θ0 = reshape(rand(d*n),(n,d));
    model0 = TD.MultivariateGammaConvolution(α0,θ0)
    data = zeros((d,N))
           
    Random.seed!(13)
    Random.rand!(model0,data)
    
    @testset "sampling a multivariate gamma convolution works." begin
        @test all(data .> 0)
    end

    Random.rand(model0)

    Random.seed!(13)
    data2 = zeros((n,N))
    for i in 1:N
        for j in 1:n
            data2[j,i] = Random.rand(Distributions.Gamma(α0[j],1))
        end
    end
    data2 = θ0'data2
    
    @testset "sampling a multivariate gamma convolution works as expected." begin
        @test data == data2
    end
    # but e could also check that samplign returns what it should return ? 
    # how to do that ? 

end



