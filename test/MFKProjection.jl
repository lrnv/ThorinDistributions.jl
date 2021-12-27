import ThorinDistributions as TD
import DoubleExponentialFormulas as DEF
using Test
using Distributions

@testset "MFKProjection.jl tests" begin

    F = Float64
    qde = DEF.QuadDE(F; h0=one(F)/8, maxlevel=10)

    @testset "Bivariate projection of f(x,y) = exp(-x-y)" begin 
        ϕ(x,p) = TD.laguerre_phi(x,p)
        f(x,y) = exp(-x-y) # the function 
        coef(fun, p) = qde(y -> qde(x -> fun(x,y)*ϕ(x,p[1])*ϕ(y,p[2]),0,Inf)[1], 0, Inf)
        @test coef(f,[0,0])[1] ≈ 1/2
        @test abs(coef(f,[1,0])[1]) ≤ eps(F)
        @test abs(coef(f,[0,1])[1]) ≤ eps(F)
        @test abs(coef(f,[1,1])[1]) ≤ eps(F)
    end
    
    @testset "Miles, Furman and kuznetsov in Float64" begin
        n = 20
        D = LogNormal(F(0),F(1))
        g = vec(TD.compute_g(D,n,qde))
        proj = TD.MFK_Projection(g,n)
        @test proj.α == [0.00901854484720319, 4.638639995014771e-5]
        @test proj.θ == [0.12909510906634175, 0.13122612690845434]
    end

    F = BigFloat
    D = LogNormal(F(0),F(1))
    qde = DEF.QuadDE(F; h0=one(F)/8, maxlevel=10)

    @testset "Miles, Fumran and kuznetsov in BigFLoat, n=10" begin
        n = 10
        g = vec(TD.compute_g(D,n,qde))
        proj = TD.MFK_Projection(g,n)
        @test proj.α == [0.003225489879746356, 0.030612760239603496, 0.0899249316838328, 0.15434895105449484, 0.21236360637036536, 0.27370969722578337, 0.35786574827597645, 0.5030569040550525, 0.8261680201005028, 2.0361698896561493]
        @test proj.θ == [0.06400949598916056, 0.14318090800168484, 0.2723072670391737, 0.48588016258877936, 0.8463539620884218, 1.4850317685534802, 2.730052993881754, 5.608408842425626, 14.826977168531004, 82.88192010967173]
    end
    @testset "Miles, Fumran and kuznetsov in BigFLoat, n=20" begin
        n=20
        g = vec(TD.compute_g(D,n,qde))
        proj = TD.MFK_Projection(g,n)

        @test proj.α == [4.237047711089287e-5, 0.000782631263408077, 0.004618206434257392, 0.014570221059751146, 0.03077896249626556, 0.05003438119515144, 0.06899011623468541, 0.08622521160094941, 0.10194681302889311, 0.1171171990990493, 0.13296720647769758, 0.1509055512072555, 0.17267484742492775, 0.2007279040378132, 0.23898609783756156, 0.294521240087835, 0.38189900974811675, 0.5369778573787414, 0.8776124011630155, 2.1224464583276936]
        @test proj.θ == [0.025483107951961903, 0.051051534441082805, 0.08605447752990605, 0.13272958025276885, 0.19403942412824055, 0.27403473215602847, 0.37814999360209933, 0.5138102382326162, 0.6915781722871679, 0.9271078407996797, 1.2445212769875846, 1.68258274592362, 2.306759480450546, 3.2346475246921838, 4.694850641671924, 7.180979645561399, 11.92731560057251, 22.78479311679837, 57.341582973133846, 310.4990752947431]
    end
end
