# Tests for TensegrityEquilibria.jl

To run tests, navigate to the projects folder and activate the project's local environment. Afterwards, simply run the command `test`.

```
julia> cd("<your_julia_home_folder>\\TesengrityEquilibria")
julia> pwd()      # Print the cursor's current location
"<your_julia_home_folder>\\TesengrityEquilibria"
julia> ]          # Pressing ] let's us enter Julia's package manager
(@v1.6) pkg> activate .
(HomotopyOpt) pkg> test
```

At the moment, this runs Tests for the animation, for the Zeeman Catastrophe Machine, the Bipyramid, the 3- and the 4-Prism.