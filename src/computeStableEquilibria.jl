using LinearAlgebra, HomotopyContinuation

#=
@input vertices::[[p_11,p_12,...], ..., [p_m1,p_m2,...]], unknownBars::[[i,j,l_ij],...], unknowncables::[[i,j,r_ij,e_ij]] with i<j
Calculate configurations in equilibrium for the given tensegrity framework
=#
function stableEquilibria(vertices::Array, unknownBars::Array, unknownCables::Array, listOfInternalVariables::Array, listOfControlParams::Array, targetParams::Array)
    assertCorrectInput(vertices, unknownBars, unknownCables, listOfInternalVariables, listOfControlParams, targetParams)

    @var delta[1:length(unknownCables)];
    B=[]; C=[]; G=[]
    for index in 1:length(unknownBars)
        l_ij=unknownBars[index][3]; p_i=vertices[Int64(unknownBars[index][1])]; p_j=vertices[Int64(unknownBars[index][2])];
        append!(B, [sum((p_i-p_j).^2)-l_ij^2])
    end

    for index in 1:length(unknownCables)
        r_ij=unknownCables[index][3]; p_i=vertices[Int64(unknownCables[index][1])]; p_j=vertices[Int64(unknownCables[index][2])]; e_ij=unknownCables[index][4];
        append!(C, [sum((p_i-p_j).^2)-delta[index]^2])
    end
    append!(G,B); append!(G,C);
    e = map(t->t[4], unknownCables); r = map(t->t[3], unknownCables);
    Q = sum(e.*(r-delta).^2/2)

    @var lambda[1:length(G)]
    L = Q + lambda'*G
    ∇L = differentiate(L, [listOfInternalVariables; delta; lambda])
    F=[]
    if(!isempty(listOfInternalVariables))
        if(!isempty(listOfControlParams))
            F = System(∇L; variables = [listOfInternalVariables; delta; lambda], parameters = listOfControlParams)
            params₀ = randn(ComplexF64, nparameters(F))
            res₀ = solve(F; target_parameters = params₀)
            S₀ = solutions(res₀)
            H = ParameterHomotopy(F, params₀, params₀)
            solver, _ = solver_startsolutions(H, S₀)
            res₀ = solve(F; target_parameters = targetParams)
            reals = filter(en->isempty(filter(x->x<0, en[length(listOfInternalVariables)+1:length(listOfInternalVariables)+length(delta)])) &&
                isLocalMinimum(listOfInternalVariables, listOfControlParams, delta, lambda, L, G)(real.(en), targetParams), real_solutions(res₀))
        else
            F = System(∇L; variables = [listOfInternalVariables; delta; lambda])
            res₀ = solve(F)
            reals = filter(en->isempty(filter(x->x<0, en[length(listOfInternalVariables)+1:length(listOfInternalVariables)+length(delta)])) &&
                isLocalMinimum(listOfInternalVariables, [], delta, lambda, L, G)(real.(en), []),real_solutions(res₀))
        end
    else
        throw(error("Internal variables need to be provided!"))
    end
    reals = map(t->t[1:length(listOfInternalVariables)], reals)
    return(reals, listOfInternalVariables)
end

# Check if the current configuration is a local minimum of Q
function isLocalMinimum(listOfInternalVariables, listOfControlParams, delta, lambda, L, G)
    ∇L = differentiate(L, [listOfInternalVariables; delta])
    if(isempty(listOfControlParams))
        HL = InterpretedSystem(System(∇L, [listOfInternalVariables; delta], lambda))
        dG = InterpretedSystem(System(G, [listOfInternalVariables; delta], lambda))
    else
        HL = InterpretedSystem(System(∇L, [listOfInternalVariables; delta], [lambda; listOfControlParams]))
        dG = InterpretedSystem(System(G, [listOfInternalVariables; delta], [lambda; listOfControlParams]))
    end

    (s, params) -> begin
      v = s[1:length(delta)+length(listOfInternalVariables)]
      q = [s[1+length(delta)+length(listOfInternalVariables):end]; params]
      W = zeros(length(delta)+length(listOfInternalVariables), length(delta)+length(listOfInternalVariables))
      jacobian!(W, HL, v, q)
      J = zeros(length(delta)+length(listOfInternalVariables)-1, length(delta)+length(listOfInternalVariables))
      jacobian!(J, dG, v, q)
      V = nullspace(J)
      all(e -> e ≥ 1e-14, eigvals(V' * W * V))
    end
end

# assert that the inputs of stableEquilibria are correct
function assertCorrectInput(vertices, unknownBars, unknownCables, listOfInternalVariables, listOfControlParams, targetParams)
    @assert(length(vertices)>1)
    D = length(vertices[1])
    foreach(vx->@assert(length(vx)==D), vertices)
    foreach(bar->@assert(length(bar)==3), unknownBars)
    foreach(cable->@assert(length(cable)==4), unknownCables)
    foreach(var->@assert(typeof(var)==Variable), listOfInternalVariables)
    foreach(par->@assert(typeof(par)==Variable), listOfControlParams)
    @assert(length(listOfControlParams)==length(targetParams))
end

@var p[1:2]
display(stableEquilibria([[1,2],[3,4],p],[[1,3,sqrt(2)]],[[2,3,1,1]],p,[],[]))

@var p[1:6]
display(stableEquilibria([p[1:3],p[4:6],[0,1,0], [sin(2*pi/3),cos(2*pi/3),0], [sin(4*pi/3),cos(4*pi/3),0]],[[1,2,1]],[[1,3,1,1],[1,4,1,1],[1,5,1,1],[2,3,1,1],[2,4,1,1],[2,5,1,1]],
    p,[],[]))

#TODO Makie plot: listOfInternalVariables->solution
