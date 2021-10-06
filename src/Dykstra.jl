module Dykstra
using LinearAlgebra
using SparseArrays
#################### LinearConstraints
struct LinearConstraint{T}
    A::SparseMatrixCSC{T,Int64}
    b::SparseVector{T, Int64}
    P::SparseMatrixCSC{T, Int64}
    q::SparseVector{T, Int64}
    rez::SparseVector{T, Int64}
end


function get_AiAA(A)
    X = A*A'
    if X == Diagonal(X)
        @assert sum(diag(X) .== 0)==0 "A*A' must not have a zero determinant"
        # @info "X is diagonal with non-zero diagonal values ! huge win."
        return A'spdiagm(1 ./ diag(X))
    else
        return A'sparse(inv(Matrix(X)))
    end
end

function LinearConstraint(A,b)
    AiAA = get_AiAA(A)
    P = -AiAA*A
    q = AiAA*b
    return LinearConstraint(sparse(A),sparse(b),sparse(P),sparse(q),spzeros(eltype(q),length(q)))
end
@inline function proj!(x,X::LinearConstraint)
    # X.rez .= x + X.P*x + X.q
    return x + X.P*x + X.q
end
@inline check(x,X::LinearConstraint) = all(abs.(X.A*x .- X.b) .<= 100*sqrt(eps(eltype(x))))


#################### DecomposedLinearConstraints
struct DecomposedLinearConstraint{T}
    A::SparseMatrixCSC{T,Int64}
    b::SparseVector{T, Int64}
    p::SparseVector{T, Int64}
    M::SparseMatrixCSC{Bool, Int64}
    q::SparseVector{T, Int64}
    rez::SparseVector{T, Int64}
end
function decompose(P;check=true)
    # check predicate and decompose: 
    if check
        @assert all([sum(unique(P[i,:]).!=0)<=1 for i in 1:size(P,1)]) "This constraint does not seem to be decomposable. Maybe use LinearConstraint instead."
    end
    b = sparse([sum(unique(P[i,:])) for i in 1:size(P,1)])
    M = sparse(BitMatrix(P .!= 0))
    return b,M
end
function DecomposedLinearConstraint(A,b;check=true)
    AiAA = get_AiAA(A)
    P = sparse(-AiAA*A)
    p,M = decompose(P,check=check)
    q = sparse(+AiAA*b)
    return DecomposedLinearConstraint(A,sparse(b),p,M,q,spzeros(eltype(q),length(q)))
end
@inline function proj!(x,X::DecomposedLinearConstraint)
    #X.rez .= x + X.q + X.p.*(X.M*x)
    return x + X.q + X.p.*(X.M*x)
end
@inline check(x,X::DecomposedLinearConstraint) = all(abs.(X.A*x .- X.b) .<= 100*sqrt(eps(eltype(x))))

#################### PositivityConstraints
struct PositivityConstraint{T} 
    rez::SparseVector{T, Int64}
    tol::T
end
function PositivityConstraint(x)
    return PositivityConstraint(spzeros(eltype(x),length(x)),sqrt(eps(eltype(x))))
end
@inline function proj!(x,X::PositivityConstraint)
    # X.rez .= x
    # SparseArrays.fkeep!(X.rez, (i,x) -> x>X.tol)
    SparseArrays.fkeep!(x, (i,x) -> x>X.tol)
    return x
end
@inline check(x,X::PositivityConstraint) = all(x.>=0)

function project!(x,ctrs;err=nothing)
    old_x = zero(x)
    err = isnothing(err) ? 10*sqrt(eps(eltype(x))) : err # warninng : this 10 should probably not be there;
    sn = [zero(x) for _ in ctrs]
    rez = zero(x)
    ctnue=!all([check(x,c) for c in ctrs])
    while ctnue
        old_x .= x
        for (i,ctr) in enumerate(ctrs)
            rez .= proj!(x-sn[i],ctr)
            sn[i] += rez - x
            x .= rez
        end
        ctnue = !all(abs.(old_x .-x) .< err)
        # display("$ctnue, $(sum(abs.(old_x .-x) .< err)) over $(length(old_x))")
    end
    return x
end

function linearize_thorin_transportation_problem(a::Vector{T},b::Vector{T}) where T
    n1 = length(a)
    n2 = length(b)
    p = (n1+1)*(n2+1)
    eq_A = zeros(T,(n1+n2,p-1))
    eq_b = [a...,b...]
    for i in 0:n2
        for j in 1:n1
            eq_A[j,j+i*(1+n1)] = 1
        end
    end
    for i in 1:n2
        eq_A[n1+i,(1:(n1+1)).+n1.+(i-1)*(n1+1)] .= 1
    end
    return eq_A,eq_b
end

end # module Dykstra