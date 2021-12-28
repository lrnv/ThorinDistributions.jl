import ThorinDistributions as TD
using Test
import Random, Distributions

@testset "MultivariateGammaConvolution.jl tests" begin 
    
    # Making data: 
    Random.seed!(13)
    N,n,d = 7,3,4
    α0 = randn(n).^2;
    θ0 = reshape(rand(d*n),(n,d));
    model0 = TD.MultivariateGammaConvolution(α0,θ0)
    data = zeros((d,N))
           
    Random.seed!(13)
    Random.rand!(model0,data)
    
    @testset "sampling a multivariate gamma convolution works." begin
        @test all(data .> 0)
    end

    Random.seed!(13)
    data2 = zeros((n,N))
    for i in 1:n
        data2[i,:] = Random.rand(Distributions.Gamma(α0[i],1),N)
    end
    data2 = θ0'data2
    
    @testset "sampling a multivariate gamma convolution works as expected." begin
        @test data == data2
    end 

end



