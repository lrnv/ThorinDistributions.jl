const MAX_M = 1000

struct PreComp{T}
    BINS::SparseArrays.SparseMatrixCSC{T,Int}
    LAGUERRE::SparseArrays.SparseMatrixCSC{T,Int}
    LAGUERRE2::SparseArrays.SparseMatrixCSC{T,Int}
    FACTS::Vector{T}
end


function bigfloat_precomputations(m)
    setprecision(2048)
    m = big(m)
    BINS = zeros(BigInt,(m,m))
    FACTS = zeros(BigInt,m)
    LAGUERRE = zeros(BigFloat,(m,m))
    LAGUERRE2 = deepcopy(LAGUERRE)
    for i in 1:m
        FACTS[i] = factorial(i-1)
    end
    for i in 1:m, j in 1:m
        BINS[j,i] = binomial(i-1,j-1)
        LAGUERRE[j,i] = BINS[j,i]/FACTS[j]*(-big(2))^(j-1)
        LAGUERRE2[j,i] = LAGUERRE[j,i] * sqrt(big(2)) * FACTS[j]
    end
    LAGUERRE = SparseArrays.sparse(LAGUERRE)
    LAGUERRE2 = SparseArrays.sparse(LAGUERRE2)
    BINS = SparseArrays.sparse(BigFloat.(BINS))
    FACTS = BigFloat.(FACTS)
    PreComp(BINS,LAGUERRE,LAGUERRE2,FACTS)
end

const big_P = bigfloat_precomputations(MAX_M)

const Precomp_Dict = Dict{Tuple{Int64,DataType},Any}((MAX_M,BigFloat)=>big_P)

"""
    get_precomp(type,m)

Query and eventualy compute and store the precomputations for laguerre things.
"""
function get_precomp(type::DataType,m::T) where {T <: Int}
    # The getter should :
    if (m,type) in keys(Precomp_Dict)
        return Precomp_Dict[(m,type)]::PreComp{type}
    end
    # if this is not the case, we will just construct it :
    if m > MAX_M
        error("You ask for too much")
    end
    P = big_P
    Precomp_Dict[(m,type)] = PreComp(type.(P.BINS[1:m,1:m]),type.(P.LAGUERRE[1:m,1:m]),type.(P.LAGUERRE2[1:m,1:m]),type.(P.FACTS[1:m]))
    return Precomp_Dict[(m,type)]::PreComp{type}
end



