module functionsForStableEquilibria

export stableEquilibria, start_demo

using LinearAlgebra, HomotopyContinuation, GLMakie, AbstractPlotting, AbstractPlotting.MakieLayout, Printf

#=
@input vertices::[[p_11,p_12,...], ..., [p_m1,p_m2,...]], unknownBars::[[i,j,l_ij],...], unknowncables::[[i,j,r_ij,e_ij]] with i<j
Calculate configurations in equilibrium for the given tensegrity framework
=#
function stableEquilibria(vertices::Array, unknownBars::Array, unknownCables::Array, listOfInternalVariables::Array{Variable,1}, listOfControlParams::Array, targetParams::Array, knownBars::Array, knownCables::Array)
    assertCorrectInput(vertices, unknownBars, unknownCables, listOfInternalVariables, listOfControlParams, targetParams, knownBars, knownCables)

    @var delta[1:length(unknownCables)]
    B, C, G = [], [], []
    for index in 1:length(unknownBars)
        l_ij=unknownBars[index][3]; p_i=vertices[Int64(unknownBars[index][1])]; p_j=vertices[Int64(unknownBars[index][2])];
        push!(B, sum((p_i-p_j).^2)-l_ij^2)
    end
    for index in 1:length(unknownCables)
        r_ij=unknownCables[index][3]; p_i=vertices[Int64(unknownCables[index][1])]; p_j=vertices[Int64(unknownCables[index][2])]; e_ij=unknownCables[index][4];
        push!(C, sum((p_i-p_j).^2)-delta[index]^2)
    end
    append!(G,B); append!(G,C);
    e = map(t->t[4], unknownCables); r = map(t->t[3], unknownCables);
    Q = sum(e.*(r-delta).^2/2)

    @var lambda[1:length(G)]
    L = Q + lambda'*G
    ∇L = differentiate(L, [listOfInternalVariables; delta; lambda])
    if(!isempty(listOfInternalVariables))
        F = isempty(listOfControlParams) ? System(∇L; variables = [listOfInternalVariables; delta; lambda]) : System(∇L; variables = [listOfInternalVariables; delta; lambda], parameters = listOfControlParams)
        params₀ = randn(ComplexF64, nparameters(F))
        try
            res₀ = isempty(listOfControlParams) ? solve(F) : solve(F; target_parameters = params₀)
            S₀ = solutions(res₀)
            H = ParameterHomotopy(F, params₀, params₀)
            solver, _ = solver_startsolutions(H, S₀)

            realSol = plotWithMakie(vertices, vcat(unknownBars, knownBars), vcat(unknownCables, knownCables), solver, S₀, listOfInternalVariables, listOfControlParams, targetParams, delta, lambda, L, G)
            return(realSol, listOfInternalVariables)
        catch e
            if(!isa(e, InterruptException) && !isa(e, BoundsError))
                println("Error ", e, " caught. Trying again...")
                stableEquilibria(vertices, unknownBars, unknownCables, listOfInternalVariables, listOfControlParams, targetParams, knownBars, knownCables)
            end
        end
    else
        throw(error("Internal variables need to be provided!"))
    end
    return(nothing)
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
# with their respective computed value. It returns the array of vertices in the form of Point2f0/Point3f0
# and the array of all other minimizers of the energy function Q, that were not yet used, in shadowPoints.
# WARNING: This cannot deal with Polynomial-type entries of the vertices yet. Only pure variables are allowed.
function arrangeArray(array, listOfInternalVariables, realSol, listOfControlParams, targetParams)
    subsArray=[]; shadowPoints=[];
    for i in 1:length(array)
        helper=[]; shadowHelper=[];
        for j in 1:length(array[i])
            if(typeof(array[i][j])!=Float64 && typeof(array[i][j])!=Int64)
                index=findfirst(en->en==array[i][j], listOfInternalVariables)
                if(index==nothing || typeof(index) == Nothing)
                    index=findfirst(en->en==array[i][j], listOfControlParams)
                    if(index==nothing || typeof(index) == Nothing)
                        throw(error("The variable in the array is neither a control nor an internal parameter."))
                    end
                    push!(helper,targetParams[index])
                    push!(shadowHelper,targetParams[index])
                else
                    push!(helper,realSol[1][index])
                    length(realSol)>1 ? push!(shadowHelper,[realSol[i][index] for i in 2:length(realSol)]) : nothing
                end
            else
                push!(helper,array[i][j])
                push!(shadowHelper,array[i][j])
            end
        end
        length(helper)==2 ? push!(subsArray,Point2f0(helper[1],helper[2])) : push!(subsArray,Point3f0(helper[1],helper[2],helper[3]))
        !all(t->typeof(t)!=Array{Float64,1}, shadowHelper) ?
            append!(shadowPoints, [length(helper)==2 ? Point2f0(typeof(shadowHelper[1])==Array{Float64,1} ? shadowHelper[1][i] : shadowHelper[1], typeof(shadowHelper[2])==Array{Float64,1} ? shadowHelper[2][i] : shadowHelper[2]) :
                Point3f0(typeof(shadowHelper[1])==Array{Float64,1} ? shadowHelper[1][i] : shadowHelper[1], typeof(shadowHelper[2])==Array{Float64,1} ? shadowHelper[2][i] : shadowHelper[2],  typeof(shadowHelper[3])==Array{Float64,1} ? shadowHelper[3][i] : shadowHelper[3])
                for i in 1:maximum([length(coord) for coord in shadowHelper])]) : length(helper)>length(shadowHelper) ? push!(shadowPoints, subsArray[end]) : nothing
    end
    return(subsArray, shadowPoints)
end

# Creates a dynamic parametric plot of the vertices (red) with corresponding bars (black) and cables (blue).
# The shadow vertices are plotted in grey. If there are no parameters given, this plot is static.
function plotWithMakie(vertices, bars, cables, solver, S₀, listOfInternalVariables, listOfControlParams, targetParams, delta, lambda, L, G)
    params=Node(targetParams)
    scene, layout = layoutscene(resolution = (1100, 900))
    ax = layout[1, 1] = length(vertices[1])==3 ? LScene(scene, width=1000, camera = cam3d!, raw = false) : LAxis(scene,width=1000)
    length(listOfControlParams)>0 ? layout[2, 1] = hbox!(LText(scene, "Control Parameters:"),width=200) : nothing

    sl=Array{Any,1}(undef, length(listOfControlParams))
    for i in 1:length(listOfControlParams)
        sl[i] = LSlider(scene, range = maximum([0.1,targetParams[i]-2]):0.1:targetParams[i]+3, startvalue = targetParams[i])
        layout[i+2, 1] = hbox!(
            LText(scene, @lift(string(string(listOfControlParams[i]), ": ", string(to_value($(sl[i].value)))))),
            sl[i],
            width=300
        )
        params = @lift begin
            params=[para for para in $params]
            params[i]=$(sl[i].value)
        end
    end

    allPossibilities = @lift begin
        target_parameters!(solver, $params)
        res₀ = solve(solver, S₀, threading = false)
        realSol = (map(t->t[1:length(listOfInternalVariables)], filter(en->isempty(filter(x->x<0, en[length(listOfInternalVariables)+1:length(listOfInternalVariables)+length(delta)])) &&
            isLocalMinimum(listOfInternalVariables, listOfControlParams, delta, lambda, L, G)(real.(en), $params), real_solutions(res₀))))
        arrangeArray(vertices, listOfInternalVariables, realSol, listOfControlParams, $params)
    end
    fixedVertices=@lift(($allPossibilities)[1])
    shadowPoints=@lift(($allPossibilities)[2])

    foreach(bar->linesegments!(ax, @lift([($fixedVertices)[Int64(bar[1])], ($fixedVertices)[Int64(bar[2])]]) ; linewidth = length(vertices[1])==2 ? 4.0 : 5.0, color=:black), bars)
    foreach(cable->linesegments!(ax, @lift([($fixedVertices)[Int64(cable[1])], ($fixedVertices)[Int64(cable[2])]]) ; color=:blue), cables)
    scatter!(ax, @lift([f for f in $shadowPoints]); color=:grey, marker = :diamond, alpha=0.3, markersize = length(vertices[1])==2 ? 13 : 45)
    scatter!(ax, @lift([f for f in $fixedVertices]); color=:red, markersize = length(vertices[1])==2 ? 13 : 45)

    if(length(vertices[1])==2)
        xlimiter = Node([Inf,-Inf]); xlimiter = @lift(computeMinMax($fixedVertices, $shadowPoints, $xlimiter, 1));
        ylimiter = Node([Inf,-Inf]); ylimiter = @lift(computeMinMax($fixedVertices, $shadowPoints, $ylimiter, 2));
        @lift(limits!(ax, FRect((($xlimiter)[1]-0.5,($ylimiter)[1]-0.5), (($xlimiter)[2]-($xlimiter)[1]+1.0,($ylimiter)[2]-($ylimiter)[1]+1.0))))
    elseif(length(vertices[1])==3)
        xlimiter = Node([Inf,-Inf]); xlimiter = @lift(computeMinMax($fixedVertices, $shadowPoints, $xlimiter, 1));
        ylimiter = Node([Inf,-Inf]); ylimiter = @lift(computeMinMax($fixedVertices, $shadowPoints, $ylimiter, 2));
        zlimiter = Node([Inf,-Inf]); zlimiter = @lift(computeMinMax($fixedVertices, $shadowPoints, $zlimiter, 3));
        #TODO set limits in 3D: How?
    end
    display(scene)
    return(fixedVertices[])
end

# This method omputes the minimal and maximal element of the position xyz a nested array (so all values of the form
# array[:][index]). Afterwards it checks if the previous minimum limiter[1] or the maximum limter[2] is beaten. If so,
# it returns the new optima.
function computeMinMax(fixedVertices,shadowPoints,limiter,xyz)
    for i in 1:length(fixedVertices)
        fixedVertices[i][xyz] > limiter[2] ? limiter[2]=fixedVertices[i][xyz] : nothing
        fixedVertices[i][xyz] < limiter[1] ? limiter[1]=fixedVertices[i][xyz] : nothing
    end
    for i in 1:length(shadowPoints)
        shadowPoints[i][xyz] > limiter[2] ? limiter[2]=shadowPoints[i][xyz] : nothing
        shadowPoints[i][xyz] < limiter[1] ? limiter[1]=shadowPoints[i][xyz] : nothing
    end
    return(limiter)
end

function start_demo()
    #Tests
    @var p[1:2] l
    display(stableEquilibria([[1,2],[3,4],p],[[1,3,2.5]],[[2,3,1,1.0]],p,[],[],[],[]))

    @var p[1:6] ell
    display(stableEquilibria([p[1:3],p[4:6],[0,1,0], [sin(2*pi/3),cos(2*pi/3),0], [sin(4*pi/3),cos(4*pi/3),0]],[[1,2,ell]],[[1,3,1,1],[1,4,1,1],[1,5,1,1],[2,3,1,1],[2,4,1,1],[2,5,1,1]],
        p,[ell],[1.0],[[3,4],[3,5],[4,5]],[]))
end

end
