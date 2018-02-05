__precompile__()

module Pincus

using Base.Cartesian, FastGaussQuadrature, Sobol, CLArrays, GPUArrays

export  gauss_hermite_tup,
        uniboxmaximize,
        threaded_uniboxmaximize,
        maximize,
        threaded_maximize,
        sobol_vec

include("opt.jl")
include("sobol.jl")

end # module
