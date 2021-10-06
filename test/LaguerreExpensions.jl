

    # data = 


    # m = 2n+1 
    # E_check = TD.empirical_coefs(c_data,(m,))


    # # reconstruction: 
    # τ = TD.thorin_moments(vec(c_data),t_star,m)
    # κ = τ .* factorial.([big(0),big.(0:m-1)...])
    # μ = TD.univ_μ_from_κ(κ)
    # A = TD.get_precomp(BigFloat,sum(m)).LAGUERRE .* sqrt(2)
    # E = A'μ[1:end-1]

    # # E should be equal to E_check.

    # and 
    #matrix_A(m)' == A