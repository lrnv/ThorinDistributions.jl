const MAX_M = 200

struct PreComp{T}
    BINS::Array{T,2}
    LAGUERRE::Array{T,2}
    FACTS::Array{T,1}
end


function bigfloat_precomputations(m)
    setprecision(2048)
    m = big(m)
    BINS = zeros(BigInt,(m,m))
    FACTS = zeros(BigInt,m)
    LAGUERRE = zeros(BigFloat,(m,m))
    for i in 1:m
        FACTS[i] = factorial(i-1)
    end
    for i in 1:m, j in 1:m
        BINS[j,i] = binomial(i-1,j-1)
        LAGUERRE[j,i] = BINS[j,i]/FACTS[j]*(-big(2))^(j-1)
    end
    BINS = BigFloat.(BINS)
    FACTS = BigFloat.(FACTS)
    PreComp(BINS,LAGUERRE,FACTS)
end

const Precomp_Dict = Dict{Tuple{Int64,DataType},Any}((MAX_M,BigFloat)=>bigfloat_precomputations(MAX_M))

"""
    get_precomp(type,m)

Query and eventualy compute and store the precomputations for laguerre things.
"""
function get_precomp(type::DataType,m::T) where {T <: Int}
    # The getter should :
    if (m,type) in keys(Precomp_Dict)
        return Precomp_Dict[(m,type)]
    end
    # if this is not the case, we will just construct it :
    if m > MAX_M
        error("You ask for too much")
    end
    P = Precomp_Dict[(MAX_M,BigFloat)]
    Precomp_Dict[(m,type)] = PreComp(type.(P.BINS[1:m,1:m]),type.(P.LAGUERRE[1:m,1:m]),type.(P.FACTS[1:m]))
    return Precomp_Dict[(m,type)]
end



