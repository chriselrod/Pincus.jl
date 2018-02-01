struct PincusResult{N,T}
    EX::NTuple{N,T}
    VX::NTuple{N,T}
    converged::Bool
    iterations::Int
end

sum_tup(x::NTuple{N}, y::NTuple{N}) where {N} = ntuple(i -> x[i] + y[i], Val{N}())
@generated zero_tuple(::Type{T}, ::Val{N}) where {N,T} = ntuple(i -> zero(T), Val{N}())
@generated function start(::Type{T}, ::Val{N}) where {N,T}
    ( zero(T), zero_tuple(T,Val{N}()), zero_tuple(T,Val{N}()) )
end
function reduction_op(x::Tuple{T,NTuple{N,T},NTuple{N,T}}, y::Tuple{T,NTuple{N,T},NTuple{N,T}}) where {N,T}
    (x[1] + y[1], x[2] .+ y[2], x[3] .+ y[3])
end

function maximize(f, grid::AbstractVector{NTuple{N,T}}, λ = one(T), max_iter::Integer = 100, ϵ = 1e-6) where {N,T}
    iterations = 0
    converged = false
    EX = zero_tuple(T, Val{N}())
    VX = ntuple(i -> one(T), Val{N}())
    function g(x, fx, EX, VX, f)
        fx = f( VX .* x .+ EX ) / fx
        xfx = x .* fx
        fx, xfx, x .* xfx
    end
    while !converged
        fx, EX, VX = mapreduce(x -> g(x, fx, EX, VX, f), sum_tup, start(T, Val{N}()))
        EX = EX ./ fx
        VX = @. sqrt( VX / fx - abs2(EX) )
        iterations += 1
        if VX < ϵ
            converged = true
        elseif iterations == max_iter
            break
        end
    end
    PincusResult(EX, VX, converged, iterations)
end

