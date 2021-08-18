using TensegrityEquilibria, Test, HomotopyContinuation

@testset "circle animation" begin
    resolution = 75
    circleset = [[[sin(2*k*pi/resolution), cos(2*k*pi/resolution)], [sin(2*k*pi/resolution+pi), cos(2*k*pi/resolution+pi)]] for k in 0:resolution]
    animateTensegrity(circleset, [[1,2]], 30)
    sleep(10)
end

@testset "Zeeman interactive" begin
    @var p[1:4];
    display(stableEquilibria([[1.0,0],[2.0,1/4],p[1:2],p[3:4]],[[1,3,0.5]],[[2,3,1/4,1/4],[3,4,1/4,1/4]],p[1:2],p[3:4],[0.0,0.0],[],[]))
    sleep(10)
end

@testset "Zeeman animation" begin
    t=0:1/15:2*pi
    timestamps=[[0.1,0.5*sin(time)-0.25] for time in t]
    @var p[1:4];
    display(stableEquilibria([[1.0,0],[2.0,1/4],p[1:2],p[3:4]],[[1,3,0.5]],[[2,3,1/4,1/4],[3,4,1/4,1/4]],p[1:2],p[3:4],[0.0,0.0],[],[],timestamps))
    sleep(10)
end

@testset "Three points catastrophe" begin
    @var p[1:2] l;
    display(stableEquilibria([[1,2],[3,4],p],[[1,3,l]],[[2,3,1,1]],p,[l],[2.5],[],[]))
    sleep(10)
end

@testset "Bipyramid interactive" begin
    @var p[1:6] ell
    display(stableEquilibria([p[1:3],p[4:6],[0,1,0],[sin(2*pi/3),cos(2*pi/3),0], [sin(4*pi/3),cos(4*pi/3),0]],
                             [[1,2,ell]],
                             [[1,3,1,1],[1,4,1,1],[1,5,1,1],[2,3,1,1],[2,4,1,1],[2,5,1,1]],
                             p,[ell],[1.0],
                             [[3,4],[3,5],[4,5]],[])
    )
    sleep(10)
end

@testset "Bipyramid static" begin
     @var p[1:6]
    display(stableEquilibria([p[1:3],p[4:6],[0,1,0],[sin(2*pi/3),cos(2*pi/3),0], [sin(4*pi/3),cos(4*pi/3),0]],
                             [[1,2,2.0]],
                             [[1,3,1,1],[1,4,1,1],[1,5,1,1],[2,3,1,1],[2,4,1,1],[2,5,1,1]],
                             p[4:6],p[1:3],[0.0,-1.0,0.0],
                             [[3,4],[3,5],[4,5]],[])
    )
    sleep(10)
end

@testset "3-Prism interactive cable constant" begin
    @var p[1:7] c
    display(stableEquilibria([[0,1,0],[sin(2*pi/3),cos(2*pi/3),0],[sin(4*pi/3),cos(4*pi/3),0],[p[1],p[2],p[7]],[p[3],p[4],p[7]],[p[5],p[6],p[7]]],
                             [[1,4,3], [2,5,3],[3,6,3]],
                             [[1,5,2.5,c],[2,6,2.5,c],[3,4,2.5,c],[4,5,1,1.0],[5,6,1,1.0],[4,6,1,1.0]],
                             p[1:6],[c,p[7]],[2.0,2.75],
                             [],
                             [[1,2],[2,3],[1,3]])
    )
    sleep(10)
end

@testset "3-Prism static" begin
    @var p[1:9]
    display(stableEquilibria([[0,1,0],[sin(2*pi/3),cos(2*pi/3),0],[sin(4*pi/3),cos(4*pi/3),0],[p[1],p[2],p[3]],[p[4],p[5],p[6]],[p[7],p[8],p[9]]],
                             [[1,4,3], [2,5,3],[3,6,3]],
                             [[1,5,2.5,1.0],[2,6,2.5,1.0],[3,4,2.5,1.0],[4,5,1,1.0],[5,6,1,1.0],[4,6,1,1.0]],
                             p[1:9],[],[],
                             [],
                             [[1,2],[2,3],[1,3]])
    )
    sleep(10)
end

@testset "3-Prism interactive point" begin
    @var p[1:9]
    display(stableEquilibria([[0,1,0],[sin(2*pi/3),cos(2*pi/3),0],[sin(4*pi/3),cos(4*pi/3),0],[p[1],p[2],p[3]],[p[4],p[5],p[6]],[p[7],p[8],p[9]]],
                             [[1,4,3], [2,5,3],[3,6,3]],
                             [[1,5,2.5,1.0],[2,6,2.5,1.0],[3,4,2.5,1.0],[4,5,1,1.0],[5,6,1,1.0],[4,6,1,1.0]],
                             p[1:6],p[7:9],[0.6053381, 0, 2.5661428],
                             [],
                             [[1,2],[2,3],[1,3]])
    )
    sleep(10)
end

@testset "4-Prism static" begin
    @var p[1:9]
    display(stableEquilibria([[0,0,0], [1,0,0], [1,1,0], [0,1,0], p[1:3], [p[4],p[5],p[3]], [p[6],p[7],p[3]], [p[8],p[9],p[3]]],
                             [[1,7,2], [2,8,2],[3,5,2],[4,6,2]],
                             [[5,6,0.5,1], [6,7,0.5,1],[7,8,0.5,1],[5,8,0.5,1], [1,5,1.,1], [2,6,1.,1], [3,7,1.,1],[4,8,1.,1]],
                             p[1:9],[],[],
                             [],
                             [[1,2], [2,3], [3,4], [1,4]])
    )
    sleep(10)
end
