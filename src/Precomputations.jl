# We could make all of this static arrays. 
# This would be a very good idea. 






const MAX_M = 30

struct PreComp{T,M,L} # L == M*M
    BINS::StaticArrays.SMatrix{M,M,T,L}
    LAGUERRE::StaticArrays.SMatrix{M,M,T,L}
    LAGUERRE2::StaticArrays.SMatrix{M,M,T,L}
    FACTS::StaticArrays.SVector{M,T}
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
    m = Int(m)
    LAGUERRE = StaticArrays.SMatrix{m,m,BigFloat,m*m}(LAGUERRE)
    LAGUERRE2 = StaticArrays.SMatrix{m,m,BigFloat,m*m}(LAGUERRE2)
    BINS = StaticArrays.SMatrix{m,m,BigFloat,m*m}(BigFloat.(BINS))
    FACTS = StaticArrays.SVector{m,BigFloat}(BigFloat.(FACTS))
    PreComp(BINS,LAGUERRE,LAGUERRE2,FACTS)
end

const Precomp_Dict = Dict{Tuple{Int64,DataType},Any}((MAX_M,BigFloat)=>bigfloat_precomputations(MAX_M))

"""
    get_precomp(type,m)

Query and eventualy compute and store the precomputations for laguerre things.
"""
function get_precomp(type::DataType,m::T) where {T <: Int}
    # The getter should :
    if (m,type) in keys(Precomp_Dict)
        return Precomp_Dict[(m,type)]::PreComp{type,Int(m),Int(m)^2}
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
    m = Int(m)
    Precomp_Dict[(m,type)] = PreComp{type,m,m*m}(
        StaticArrays.SMatrix{m,m,type,m*m}(type.(P.BINS[1:m,1:m])),
        StaticArrays.SMatrix{m,m,type,m*m}(type.(P.LAGUERRE[1:m,1:m])),
        StaticArrays.SMatrix{m,m,type,m*m}(type.(P.LAGUERRE2[1:m,1:m])),
        StaticArrays.SVector{m,type}(type.(P.FACTS[1:m]))
    )
    return Precomp_Dict[(m,type)]::PreComp{type,m,m*m}
end





