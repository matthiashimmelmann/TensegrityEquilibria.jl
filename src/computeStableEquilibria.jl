using LinearAlgebra, HomotopyContinuation, Makie

#=
@input vertices::[[p_11,p_12,...], ..., [p_m1,p_m2,...]], unknownBars::[[i,j,l_ij],...], unknowncables::[[i,j,r_ij,e_ij]] with i<j
Calculate configurations in equilibrium for the given tensegrity framework
=#
function stableEquilibria(vertices::Array, unknownBars::Array, unknownCables::Array, listOfInternalVariables::Array{Variable,1}, listOfControlParams::Array, targetParams::Array, knownBars::Array, knownCables::Array)
    assertCorrectInput(vertices, unknownBars, unknownCables, listOfInternalVariables, listOfControlParams, targetParams, knownBars, knownCables)

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
            target_parameters!(solver, targetParams)
            res₀ = solve(solver, S₀, threading = false)
            realSol = filter(en->isempty(filter(x->x<0, en[length(listOfInternalVariables)+1:length(listOfInternalVariables)+length(delta)])) &&
                isLocalMinimum(listOfInternalVariables, listOfControlParams, delta, lambda, L, G)(real.(en), targetParams), real_solutions(res₀))
        else
            F = System(∇L; variables = [listOfInternalVariables; delta; lambda])
            res₀ = solve(F)
            realSol = filter(en->isempty(filter(x->x<0, en[length(listOfInternalVariables)+1:length(listOfInternalVariables)+length(delta)])) &&
                isLocalMinimum(listOfInternalVariables, [], delta, lambda, L, G)(real.(en), []),real_solutions(res₀))
        end
    else
        throw(error("Internal variables need to be provided!"))
    end
    realSol = map(t->t[1:length(listOfInternalVariables)], realSol)

    vertices = clearArrayOfVariables(vertices, listOfInternalVariables, realSol, listOfControlParams, targetParams)
    bars = vcat(unknownBars, knownBars)
    cables = vcat(unknownCables, knownCables)
    plotWithMakie(vertices, bars, cables)
    return(realSol, listOfInternalVariables)
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
function assertCorrectInput(vertices, unknownBars, unknownCables, listOfInternalVariables, listOfControlParams, targetParams, knownBars, knownCables)
    @assert(length(vertices)>1)
    D = length(vertices[1])
    foreach(vx->@assert(length(vx)==D), vertices)
    foreach(bar->@assert(length(bar)==3), unknownBars)
    foreach(cable->@assert(length(cable)==4), unknownCables)
    foreach(bar->@assert(length(bar)==2), knownBars)
    foreach(cable->@assert(length(cable)==2), knownCables)
    foreach(var->@assert(typeof(var)==Variable), listOfInternalVariables)
    foreach(par->@assert(typeof(par)==Variable), listOfControlParams)
    @assert(length(listOfControlParams)==length(targetParams))
end

# This function extracts the position of the parameters in the vertices and replaces them
# with their respective computed value. WARNING: This cannot deal with Polynomial-type entries
# of the vertices yet. Only pure variables are allowed.
function clearArrayOfVariables(array, listOfInternalVariables, realSol, listOfControlParams, targetParams)
    subsArray=[]
    for i in 1:length(array)
        append!(subsArray,[[]])
        for j in 1:length(array[i])
            if(typeof(array[i][j])!=Float64 && typeof(array[i][j])!=Int64)
                index=findfirst(en->en==array[i][j], listOfInternalVariables)
                if(index==nothing || typeof(index) == Nothing || typeof(index) == Void)
                    index=findfirst(en->en==array[i][j], listOfControlParams)
                    if(index==nothing || typeof(index) == Nothing || typeof(index) == Void)
                        throw(error("The variable in the array is neither a control nor an internal parameter."))
                    end
                    append!(subsArray[i],[targetParams[index]])
                else
                    append!(subsArray[i],[realSol[1][index]])
                end
            else
                append!(subsArray[i],[array[i][j]])
            end
        end
    end
    return(subsArray)
end

# Creates a static, non-parametric plot of the vertices (grey) with corresponding bars (black) and cables (blue).
function plotWithMakie(vertices::Array, bars::Array, cables::Array)
    scene = Scene()
    D = length(vertices[1])
    if(D==2)
        x = [vx[1] for vx in vertices]; y = [vx[2] for vx in vertices];
        scene=Makie.scatter(x,y,markersize=25,color=:grey)
        for line in bars
            Makie.lines!([vertices[Int64(line[1])][1],vertices[Int64(line[2])][1]], [vertices[Int64(line[1])][2],vertices[Int64(line[2])][2]], linewidth=5,color=:black)
        end
        for line in cables
            Makie.lines!([vertices[Int64(line[1])][1],vertices[Int64(line[2])][1]], [vertices[Int64(line[1])][2],vertices[Int64(line[2])][2]], color=:blue)
        end
    elseif(D==3)
        x = [vx[1] for vx in vertices]; y = [vx[2] for vx in vertices]; z = [vx[3] for vx in vertices]
        scene=Makie.scatter(x,y,z,markersize=25,color=:grey)
        for line in bars
            Makie.lines!([vertices[Int64(line[1])][1],vertices[Int64(line[2])][1]], [vertices[Int64(line[1])][2],vertices[Int64(line[2])][2]], [vertices[Int64(line[1])][3],vertices[Int64(line[2])][3]], linewidth=7, color=:black)
        end
        for line in cables
            Makie.lines!([vertices[Int64(line[1])][1],vertices[Int64(line[2])][1]], [vertices[Int64(line[1])][2],vertices[Int64(line[2])][2]], [vertices[Int64(line[1])][3],vertices[Int64(line[2])][3]], color=:blue)
        end
    else
        throw(error("Plot only supported for 2D and 3D Frameworks."))
    end
    display(scene)
end

@var p[1:2]
display(stableEquilibria([[1,2],[3,4],p],[[1,3,sqrt(2)]],[[2,3,1,1]],p,[],[],[],[]))

@var p[1:6] ell
display(stableEquilibria([p[1:3],p[4:6],[0,1,0], [sin(2*pi/3),cos(2*pi/3),0], [sin(4*pi/3),cos(4*pi/3),0]],[[1,2,ell]],[[1,3,1,1],[1,4,1,1],[1,5,1,1],[2,3,1,1],[2,4,1,1],[2,5,1,1]],
    p,[ell],[1],[[3,4],[3,5],[4,5]],[]))

#TODO Makie plot: listOfInternalVariables->solution
