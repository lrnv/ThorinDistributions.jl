const MAX_M = 100

struct PreComp{T}
    BINS::SparseArrays.SparseMatrixCSC{T,Int}
    LAGUERRE::SparseArrays.SparseMatrixCSC{T,Int}
    LAGUERRE_NOTSPARSE::Matrix{T}
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
    LAGUERRE_NOTSPARSE = deepcopy(LAGUERRE)
    LAGUERRE = SparseArrays.sparse(LAGUERRE)
    LAGUERRE2 = SparseArrays.sparse(LAGUERRE2)
    BINS = SparseArrays.sparse(BigFloat.(BINS))
    FACTS = BigFloat.(FACTS)
    PreComp(BINS,LAGUERRE,LAGUERRE_NOTSPARSE,LAGUERRE2,FACTS)
end

const Precomp_Dict = Dict{Tuple{Int64,DataType},Any}((MAX_M,BigFloat)=>bigfloat_precomputations(MAX_M))

"""
    get_precomp(type,m)

Query and eventualy compute and store the precomputations for laguerre things.
"""
function get_precomp(type::DataType,m::T) where {T <: Int}
    # The getter should :
    if (m,type) in keys(Precomp_Dict)
        return Precomp_Dict[(m,type)]::PreComp{type}
    end
    # # if this is not the case, we will just construct it :
    # if m > MAX_M
    #     error("You ask for too much")
    # end
    # P = big_P
    if m > MAX_M
        Precomp_Dict[(m,BigFloat)] = bigfloat_precomputations(m)
    end
    P = Precomp_Dict[max(m,MAX_M),BigFloat]
    Precomp_Dict[(m,type)] = PreComp(
        type.(P.BINS[1:m,1:m]),
        type.(P.LAGUERRE[1:m,1:m]),
        type.(P.LAGUERRE_NOTSPARSE[1:m,1:m]),
        type.(P.LAGUERRE2[1:m,1:m]),
        type.(P.FACTS[1:m])
    )
    return Precomp_Dict[(m,type)]::PreComp{type}
end





