module TensegrityEquilibria

export stableEquilibria,
       start_demo,
       animateTensegrity

import LinearAlgebra: norm, nullspace, zeros, eigvals
import HomotopyContinuation: @var, solve, differentiate, System, InterpretedSystem, randn, target_parameters!, real_solutions, nparameters, jacobian!, monodromy_solve, Variable, solutions, ParameterHomotopy, solver_startsolutions, parameters, variables
import GLMakie: scatter!, Node, @lift, limits!, linesegments!, record, Point3f0, on, Point2f0, Point, FRect, Scene, cam3d!, xlims!, ylims!, zlims!
import GLMakie.AbstractPlotting: layoutscene, labelslidergrid!, Box, Label, LScene, MakieLayout, set_close_to!, Axis
import Printf

#=
@input vertices::[[p_11,p_12,...], ..., [p_m1,p_m2,...]], unknownBars::[[i,j,l_ij],...] non-redundant list, unknowncables::[[i,j,r_ij,e_ij]] non-redundant list,
listOfInternalVariables::[a,...], listOfControlParams::[a,...], targetParams::[a,...], knownBars::[[i,j],...] for plotting, knownCables::[[i,j],...] for plotting.
If an animation is wished, use the optional argument timestamps::[[q_11,q_12,...],...] to provide a path (given by points) in the space listOfControlParams.
Calculate configurations in equilibrium for the given tensegrity framework
=#
function stableEquilibria(vertices::Array, unknownBars::Array, unknownCables::Array, listOfInternalVariables::Array{Variable,1}, listOfControlParams::Array, targetParams::Array, knownBars::Array, knownCables::Array, timestamps=[])
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
    catastropheWitness = catastrophePoints([entry for entry in vertices], listOfInternalVariables, listOfControlParams, targetParams, unknownCables, unknownBars)
    if(!isempty(listOfInternalVariables))
        F = isempty(listOfControlParams) ? System(∇L; variables = [listOfInternalVariables; delta; lambda]) : System(∇L; variables = [listOfInternalVariables; delta; lambda], parameters = listOfControlParams)
        params₀ = randn(ComplexF64, nparameters(F))
        #TODO compute a local solution via gradient descent
        res₀ = isempty(listOfControlParams) ? solve(F) : solve(F; target_parameters = params₀)

        S₀ = solutions(res₀)
        H = ParameterHomotopy(F, params₀, params₀)
        solver, _ = solver_startsolutions(H, S₀)
        if(isempty(timestamps))
            # produces a non-animate, interactive plots
            realSol = plotWithMakie(vertices, unknownBars, knownBars, unknownCables, knownCables, solver, S₀, listOfInternalVariables, listOfControlParams, targetParams, delta, lambda, L, G, catastropheWitness)
            return(realSol, listOfInternalVariables)
        else
            # produces an animation
            realSol = animateSolvedFramework(vertices, unknownBars, knownBars, unknownCables, knownCables, solver, S₀, listOfInternalVariables, listOfControlParams, targetParams, delta, lambda, L, G, catastropheWitness, timestamps)
            return(realSol, listOfInternalVariables)
        end
    else
        throw(error("Internal variables need to be provided!"))
    end
    return(nothing)
end

#= Check if the input configuration (a critical point) is a local minimum of the energy function or not.
=#
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

#= assert that the inputs of stableEquilibria are as expected
=#
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

#= This function extracts the position of the parameters in the vertices and replaces them
 with their respective computed value. It returns the array of vertices in the form of Point2f0/Point3f0
 and the array of all other minimizers of the energy function Q, that were not yet used, in shadowPoints.
 WARNING: This cannot deal with Polynomial-type entries of the vertices yet. Only pure variables are allowed.=#
function arrangeArray(array, listOfInternalVariables, realSol, listOfControlParams, targetParams)
    #TODO add polynomial substitution for coordinates on a curve (polynomials)
    subsArray=[[] for sol in realSol]
    for i in 1:length(array)
        helper=[];
        for j in 1:length(array[i])
            # We check, if the current array element is a number or a variable
            if(typeof(array[i][j])!=Float64 && typeof(array[i][j])!=Int64 && typeof(array[i][j])!=ComplexF64)
                index=findfirst(en->en==array[i][j], listOfInternalVariables)
                if(index==nothing || typeof(index) == Nothing)
                    index=findfirst(en->en==array[i][j], listOfControlParams)
                    if(index==nothing || typeof(index) == Nothing)
                        throw(error("The variable in the array is neither a control nor an internal parameter."))
                    end
                    push!(helper,targetParams[index])
                else
                    push!(helper,[realSol[i][index] for i in 1:length(realSol)])
                end
            else
                push!(helper,array[i][j])
            end
        end
        # All possible configurations are added to an array. Later, one is chosen to be the vertex to be displayed and the others are chosen to be shadow points.
        for i in 1:length(realSol)
            if length(helper)==2
                push!(subsArray[i], Point2f0(typeof(helper[1])==Array{Float64,1} ? helper[1][i] : helper[1], typeof(helper[2])==Array{Float64,1} ? helper[2][i] : helper[2]))
            else
                push!(subsArray[i], Point3f0(typeof(helper[1])==Array{Float64,1} ? helper[1][i] : helper[1], typeof(helper[2])==Array{Float64,1} ? helper[2][i] : helper[2], typeof(helper[3])==Array{Float64,1} ? helper[3][i] : helper[3]))
            end
        end
    end
    return(subsArray)
end

#= Creates a dynamic parametric plot of the vertices (red) with corresponding bars (black) and cables (blue).
The shadow vertices are plotted in grey. If there are no parameters given, this plot is static.
It allows for swapping between the different local minima of the energy function with the 'n' key.=#
function plotWithMakie(vertices, unknownBars, knownBars, unknownCables, knownCables, solver, S₀, listOfInternalVariables, listOfControlParams, targetParams, delta, lambda, L, G, catastrophePoints)
    bars=vcat(unknownBars,knownBars); cables=vcat(unknownCables,knownCables)
    # Make the variable params interactive.
    params=Node(targetParams)

    scene, layout = layoutscene(resolution = (1400, 850))
    ax = layout[1:4, 1] = length(vertices[1])==3 ? LScene(scene, width=750, height=750, camera = cam3d!, raw = false) : MakieLayout.Axis(scene,width=750,height=750)

    # Initialization of the scene's layout.
    layout[1:4, 2] = Box(scene, color = :white, strokecolor = :transparent, width=50)
    layout[1, 3] = Box(scene, color = :white, strokecolor = :transparent, height=50)
    layout[2, 3] = Box(scene, color = (:white, 0.1), strokecolor = :red, height=80, width=400)
    layout[2, 3] = Label(scene, "Press the 'n' Key to iterate through\nthe different vertex configurations", textsize = 20, halign=:center, valign=:center, color=:teal)
    layout[3, 3] = Box(scene, color = (:white, 0.1), strokecolor = :transparent, height=100)
    layout[3, 3] = Label(scene, "Control Parameters:", textsize = 20, halign=:left, valign=:bottom)

    # Adding sliders for interactively choosing the parameters during runtime.
    if length(listOfControlParams)>0
        lsgrid = labelslidergrid!(
            scene,
            [string(string(para),": ") for para in listOfControlParams],
            [targetParams[i]-2:0.05:targetParams[i]+2 for i in 1:length(targetParams)];
            formats = x -> "$(x==(round(x, digits = 1)) ? string(string(x),"0") : string(x))",
            width = 400,
            valign=:top
        )
        layout[4,3]=lsgrid.layout
        for i in 1:length(listOfControlParams)
            set_close_to!(lsgrid.sliders[i], targetParams[i])
            params = @lift begin
                param = ($params)
                param[i]=$(lsgrid.sliders[i].value)
                return(param)
            end
        end
    end

    # Upon change of params, recalculate the possible vertex configurations of this system
    allVertices=@lift begin
        target_parameters!(solver, $params)
        res₀ = solve(solver, S₀, threading = true)
        realSol = (map(t->t[1:length(listOfInternalVariables)+length(delta)], filter(en->isempty(filter(x->x<0, en[length(listOfInternalVariables)+1:length(listOfInternalVariables)+length(delta)])) &&
            isLocalMinimum(listOfInternalVariables, listOfControlParams, delta, lambda, L, G)(real.(en), $params), real_solutions(res₀))))
        arrangeArray(vertices, listOfInternalVariables, [sol[1:length(listOfInternalVariables)] for sol in realSol], listOfControlParams, $params)
    end

    # Make the index of the current configuration in the space of all possible configurations `allVertices` interactively choosable. Change it during runtime by pressing 'n'
    currentIndex = Node(1)
    on(scene.events.unicode_input) do input
        'n' in input ? (currentIndex[]-1==0 ? currentIndex[]=length(allVertices[]) : currentIndex[]=currentIndex[]-1) : nothing
    end
    fixedVertices=@lift(($allVertices)[$(currentIndex) > length($allVertices) ? 1 : $(currentIndex)])
    shadowPoints=@lift(vcat(($allVertices)[1:end]...))

    # Plot all given edges and vertices. They are modified upon registering change in `allVertices`, which is equivalent to change in params.
    foreach(bar->linesegments!(ax, @lift([($fixedVertices)[Int64(bar[1])], ($fixedVertices)[Int64(bar[2])]]) ; linewidth = length(vertices[1])==2 ? 4.0 : 5.0, color=:black), bars)
    foreach(cable->linesegments!(ax, @lift([($fixedVertices)[Int64(cable[1])], ($fixedVertices)[Int64(cable[2])]]) ; color=:blue), cables)
    scatter!(ax, @lift([f for f in $shadowPoints]); color=:goldenrod3, marker = :diamond, alpha=0.1, markersize = length(vertices[1])==2 ? 12 : 75)
    scatter!(ax, @lift([f for f in $fixedVertices]); color=:red, markersize = length(vertices[1])==2 ? 12 : 75)
    #mesh!(ax, catastrophePoints, color=:lightgrey, alpha=0.15)
    #meshing(ax, catastrophePoints, 5)
    !isempty(catastrophePoints) ? scatter!(ax, catastrophePoints; alpha=0.1, markersize=0.5, color=:lightgrey) : nothing
    if(length(vertices[1])==2)
        xlimiter = Node([Inf,-Inf]); xlimiter = @lift(computeMinMax($fixedVertices, $shadowPoints, $xlimiter, 1));
        ylimiter = Node([Inf,-Inf]); ylimiter = @lift(computeMinMax($fixedVertices, $shadowPoints, $ylimiter, 2));
        @lift(limits!(ax, FRect((($xlimiter)[1]-0.5,($ylimiter)[1]-0.5), (($xlimiter)[2]-($xlimiter)[1]+1.0,($ylimiter)[2]-($ylimiter)[1]+1.0))))
    end
    display(scene)
    return(fixedVertices[])
end

#= This method omputes the minimal and maximal element of the position xyz a nested array (so all values of the form
array[:][index]). Afterwards it checks if the previous minimum limiter[1] or the maximum limter[2] is beaten. If so,
it returns the new optima.=#
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

function meshing(scene, points, density)
    for i in 1:length(points)
        v = [norm(points[i]-points[j]) for j in 1:length(points)]
        idx = filter(j->j!=i,partialsortperm(v, 1:density+1; rev=false))
        foreach(j->linesegments!(scene, [points[i],points[j]]; linewidth = 0.7, alpha=0.2, color=:lightgrey), idx)
    end
end

#= The method catastrophePoints computes a witness set of the catastrophe discriminant using monodromy loops
 If the computation fails for some reason, it returns nothing, so the plot can still be displayed.=#
function catastrophePoints(vertices, internalVariables, controlParameters, targetParams, unknownCables, unknownBars)
    cp = let
        #TODO find control node
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
        ∇L = differentiate(L, [internalVariables; delta; lambda])
        @var v[1:length(∇L)]
        α = randn(length(∇L))
        J_L_v = [differentiate(∇L, [internalVariables; delta; lambda]) * v; α'v - 1]
        @var a[1:length(controlParameters),1:length(controlParameters)-1] b[1:length(controlParameters)-1]
        L1 = a' * controlParameters + b
        try
            P = System(
              [∇L; J_L_v; L1];
              variables = [controlParameters; internalVariables; delta; lambda; v],
              parameters = [collect(Iterators.flatten(a)); b]
            )
            res = monodromy_solve(P)
            #res=solve(P, target_parameters=randn(ComplexF64, nparameters(P)))
            rand_lin_space = let
                () -> randn(nparameters(P))
            end
            N = 500
            alg_catastrophe_points = solve(
                P,
                solutions(res),
                start_parameters = parameters(res),#startParams,
                target_parameters = [rand_lin_space() for i = 1:N],
                transform_result = (r, p) -> real_solutions(r),
                flatten = true
            )
            filter!(p -> all(k->k ≥ 0, p[length(controlParameters)+length(internalVariables)+1:length(controlParameters)+length(internalVariables)+length(delta)]), alg_catastrophe_points);
            cp = map(p -> length(vertices[1])==2 ? Point2f0(p[1], p[2]) : Point3f0(p[1], p[2], p[3]), alg_catastrophe_points)
            return(cp)
        catch e
            display(e)
        end
        return([])
    end
    return(cp)
end

#= This method allows for the animation of a tensegrity framework. Given a path in parameter space defined by point samples, it animates the movement
 of the energy functions' minima along this path.=#
function animateSolvedFramework(vertices, unknownBars, knownBars, unknownCables, knownCables, solver, S₀, listOfInternalVariables, listOfControlParams, targetParams, delta, lambda, L, G, cp, timestamps)
    bars=vcat(unknownBars,knownBars); cables=vcat(unknownCables,knownCables)
    params=Node(targetParams)
    scene, layout = layoutscene(resolution = (850, 850))
    ax = layout[1:length(listOfControlParams)+2, 1] = length(vertices[1])==3 ? LScene(scene, width=750, height=750, camera = cam3d!, raw = false) : Axis(scene,width=750,height=750)

    allVertices=@lift begin
        target_parameters!(solver, $params)
        res₀ = solve(solver, S₀, threading = true)
        realSol = (map(t->t[1:length(listOfInternalVariables)], filter(en->isempty(filter(x->x<0, en[length(listOfInternalVariables)+1:length(listOfInternalVariables)+length(delta)])) &&
            isLocalMinimum(listOfInternalVariables, listOfControlParams, delta, lambda, L, G)(real.(en), $params), real_solutions(res₀))))
        arrangeArray(vertices, listOfInternalVariables, realSol, listOfControlParams, $params)
    end
    currentIndex=Node(1)
    fixedVertices=@lift(($allVertices)[$(currentIndex) > length($allVertices) ? 1 : $(currentIndex)])
    shadowPoints=@lift(vcat(($allVertices)[1:end]...))

    foreach(bar->linesegments!(ax, @lift([($fixedVertices)[Int64(bar[1])], ($fixedVertices)[Int64(bar[2])]]) ; linewidth = length(vertices[1])==2 ? 4.0 : 5.0, color=:black), bars)
    foreach(cable->linesegments!(ax, @lift([($fixedVertices)[Int64(cable[1])], ($fixedVertices)[Int64(cable[2])]]) ; color=:blue), cables)
    scatter!(ax, @lift([f for f in $shadowPoints]); color=:goldenrod3, marker = :diamond, alpha=0.1, markersize = length(vertices[1])==2 ? 12 : 75)
    scatter!(ax, @lift([f for f in $fixedVertices]); color=:red, markersize = length(vertices[1])==2 ? 12 : 75)
    scatter!(ax, cp; alpha=0.1, markersize=0.5, color=:teal)
    if(length(vertices[1])==2)
        xlimiter = Node([Inf,-Inf]); xlimiter = @lift(computeMinMax($fixedVertices, $shadowPoints, $xlimiter, 1));
        ylimiter = Node([Inf,-Inf]); ylimiter = @lift(computeMinMax($fixedVertices, $shadowPoints, $ylimiter, 2));
        @lift(limits!(ax, FRect((($xlimiter)[1]-0.5,($ylimiter)[1]-0.5), (($xlimiter)[2]-($xlimiter)[1]+1.0,($ylimiter)[2]-($ylimiter)[1]+1.0))))
    end

    # Animate the framework by giving params different values and thus triggering change in the framework.
    record(scene, "time_animation.gif", timestamps; framerate = 15) do t
        params[] = t
    end
    display(scene)
    return(fixedVertices[])
end

#= transform a list of vertices into a list of points.
=#
function to_point(listOfVertices)
    if(length(listOfVertices[1])==2)
        return([Point2f0([vx[1],vx[2]]) for vx in listOfVertices])
    else
        return([Point3f0([vx[1],vx[2],vx[3]]) for vx in listOfVertices])
    end
end

#= Given a list of lists of vertices (on a parameter path) as a list of Point2f0 or Point3f0 and a list of edges ([[i,j], ...]). fr is the preferred framerate
=#
function animateTensegrity(listOfVertexSets::Array, listOfEdges::Array, fr::Int)
    currentConfiguration = Node(listOfVertexSets[1])
    scene=Scene()
    scatter!(scene, @lift([f for f in to_point($currentConfiguration)]))
    for edge in listOfEdges
        linesegments!(scene, @lift([to_point($currentConfiguration)[edge[1]], to_point($currentConfiguration)[edge[2]]]))
    end

    timestamps = [i for i in 1:length(listOfVertexSets)]
    record(scene, "animatedTensegrity.gif", timestamps; framerate = fr) do i
        currentConfiguration[] = listOfVertexSets[i]
    end
    display(scene)
end

function start_demo(whichTests::Array)
    #Tests
    if(-1 in whichTests)
        animateTensegrity([[[0,1],[1,0]], [[0,0],[1,1]],], [[1,2]], 1)
    end

    if(0 in whichTests)
        @var p[1:4];
        display(stableEquilibria([[1.0,0],[2.0,1/4],p[1:2],p[3:4]],[[1,3,0.5]],[[2,3,1/4,1/4],[3,4,1/4,1/4]],p[1:2],p[3:4],[0.0,0.0],[],[]))
        sleep(5)
    end

    if(0.1 in whichTests)
        t=0:1/15:2*pi
        timestamps=[[0.1,0.5*sin(time)-0.25] for time in t]
        @var p[1:4];
        display(stableEquilibria([[1.0,0],[2.0,1/4],p[1:2],p[3:4]],[[1,3,0.5]],[[2,3,1/4,1/4],[3,4,1/4,1/4]],p[1:2],p[3:4],[0.0,0.0],[],[],timestamps))
        sleep(5)
    end

    if(1 in whichTests)
        @var p[1:2] l;
        display(stableEquilibria([[1,2],[3,4],p],[[1,3,l]],[[2,3,1,1]],p,[l],[2.5],[],[]))
        sleep(5)
    end

    if(2 in whichTests)
        @var p[1:6] ell
        display(stableEquilibria([p[1:3],p[4:6],[0,1,0],[sin(2*pi/3),cos(2*pi/3),0], [sin(4*pi/3),cos(4*pi/3),0]],
                                 [[1,2,ell]],
                                 [[1,3,1,1],[1,4,1,1],[1,5,1,1],[2,3,1,1],[2,4,1,1],[2,5,1,1]],
                                 p,[ell],[1.0],
                                 [[3,4],[3,5],[4,5]],[])
        )
        sleep(5)
    end

    if(2.1 in whichTests)
        @var p[1:6]
        display(stableEquilibria([p[1:3],p[4:6],[0,1,0],[sin(2*pi/3),cos(2*pi/3),0], [sin(4*pi/3),cos(4*pi/3),0]],
                                 [[1,2,2.0]],
                                 [[1,3,1,1],[1,4,1,1],[1,5,1,1],[2,3,1,1],[2,4,1,1],[2,5,1,1]],
                                 p[4:6],p[1:3],[0.0,-1.0,0.0],
                                 [[3,4],[3,5],[4,5]],[])
        )
        sleep(5)
    end

    if(3 in whichTests)
        @var p[1:7] c
        display(stableEquilibria([[0,1,0],[sin(2*pi/3),cos(2*pi/3),0],[sin(4*pi/3),cos(4*pi/3),0],[p[1],p[2],p[7]],[p[3],p[4],p[7]],[p[5],p[6],p[7]]],
                                 [[1,4,3], [2,5,3],[3,6,3]],
                                 [[1,5,2.5,c],[2,6,2.5,c],[3,4,2.5,c],[4,5,1,1.0],[5,6,1,1.0],[4,6,1,1.0]],
                                 p[1:6],[c,p[7]],[2.0,2.75],
                                 [],
                                 [[1,2],[2,3],[1,3]])
        )
        sleep(5)
    end

    if(3.1 in whichTests)
        @var p[1:9]
        display(stableEquilibria([[0,1,0],[sin(2*pi/3),cos(2*pi/3),0],[sin(4*pi/3),cos(4*pi/3),0],[p[1],p[2],p[3]],[p[4],p[5],p[6]],[p[7],p[8],p[9]]],
                                 [[1,4,3], [2,5,3],[3,6,3]],
                                 [[1,5,2.5,1.0],[2,6,2.5,1.0],[3,4,2.5,1.0],[4,5,1,1.0],[5,6,1,1.0],[4,6,1,1.0]],
                                 p[1:9],[],[],
                                 [],
                                 [[1,2],[2,3],[1,3]])
        )
        sleep(5)
    end

    if(3.2 in whichTests)
        @var p[1:9]
        display(stableEquilibria([[0,1,0],[sin(2*pi/3),cos(2*pi/3),0],[sin(4*pi/3),cos(4*pi/3),0],[p[1],p[2],p[3]],[p[4],p[5],p[6]],[p[7],p[8],p[9]]],
                                 [[1,4,3], [2,5,3],[3,6,3]],
                                 [[1,5,2.5,1.0],[2,6,2.5,1.0],[3,4,2.5,1.0],[4,5,1,1.0],[5,6,1,1.0],[4,6,1,1.0]],
                                 p[1:6],p[7:9],[0.6053381, 0, 2.5661428],
                                 [],
                                 [[1,2],[2,3],[1,3]])
        )
        sleep(5)
    end

    if(4 in whichTests)
        @var p[1:9]
        display(stableEquilibria([[0,0,0], [1,0,0], [1,1,0], [0,1,0], p[1:3], [p[4],p[5],p[3]], [p[6],p[7],p[3]], [p[8],p[9],p[3]]],
                                 [[1,7,2], [2,8,2],[3,5,2],[4,6,2]],
                                 [[5,6,0.5,1], [6,7,0.5,1],[7,8,0.5,1],[5,8,0.5,1], [1,5,1.,1], [2,6,1.,1], [3,7,1.,1],[4,8,1.,1]],
                                 p[1:9],[],[],
                                 [],
                                 [[1,2], [2,3], [3,4], [1,4]])
        )
        sleep(5)
    end
end

end
