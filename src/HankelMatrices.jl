function hankel_operators(m,a::T,b::T) where T
    if iseven(m)
        # m = 2n
        n = Int(m//2)
        H_low = zeros(T,(n+1,n+1,m+1))
        H_up = zeros(T,(n,n,m+1))
        for i in 1:n+1
            for j in 1:n+1
                H_low[i,j,i+j-1] = 1
                if (i <= n) & (j <= n)
                    H_up[i,j,i+j] = a+b
                    H_up[i,j,i+j+1] = -1
                    H_up[i,j,i+j-1] = -a*b
                end
            end
        end
    else
        # m = 2n+1
        n = Int((m-1)//2)
        H_low = zeros(T,(n+1,n+1,m+1))
        H_up = zeros(T,(n+1,n+1,m+1))
        for i in 1:n+1
            for j in 1:n+1
                H_low[i,j,i+j] = 1
                H_low[i,j,i+j-1] = -a
                H_up[i,j,i+j-1] = b
                H_up[i,j,i+j] = -1
            end
        end
    end
    return H_low, H_up
end
function get_hankels(s,a,b)
    m = length(s)-1
    H_low, H_up = hankel_operators(m,a,b)
    s_reshaped = reshape(s,(1,1,length(s)))
    return dropdims(sum(H_low .*s_reshaped, dims=3),dims=3),dropdims(sum(H_up .*s_reshaped, dims=3),dims=3)
end
vec_H(H) = reshape(H,(size(H,1)*size(H,2),size(H,3)))
check_moment_sequence(x,a,b) = LinearAlgebra.isposdef.(get_hankels(x,a,b))
