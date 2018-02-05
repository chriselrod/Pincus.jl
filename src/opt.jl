using Compat

struct PincusResult{N,T}
    EX::NTuple{N,T}
    VX::NTuple{N,T}
    converged::Bool
    iterations::Int
end

sum_tup(x::NTuple{N,T}, y::NTuple{N,T}) where {N,T} = ntuple(i -> x[i] + y[i], Val{N}())
function sum_tups(x::Tuple{T,NTuple{N,T},NTuple{N,T}}, y::Tuple{T,NTuple{N,T},NTuple{N,T}}) where {N,T}
    x[1]+y[1], sum_tup(x[2], y[2]), sum_tup(x[3], y[3])
end
@generated zero_tuple(::Type{T}, ::Val{N}) where {N,T} = ntuple(i -> zero(T), Val{N}())
@generated function init_reduce(::Type{T}, ::Val{N}) where {N,T}
    ( zero(T), zero_tuple(T,Val{N}()), zero_tuple(T,Val{N}()) )
end
function reduction_op(x::Tuple{T,NTuple{N,T},NTuple{N,T}}, y::Tuple{T,NTuple{N,T},NTuple{N,T}}) where {N,T}
    (x[1] + y[1], x[2] .+ y[2], x[3] .+ y[3])
end


function maximize(f, grid::AbstractVector{Tuple{T,NTuple{N,T}}}, ::Val{verbose} = Val{true}(), λ = T(0.1), max_iter::Integer = 30, ϵ = 1e-3, step = naive_step) where {N,T, verbose}
    iterations = 0
    converged = false
    EX::NTuple{N,T} = zero_tuple(T, Val{N}())
    VX::NTuple{N,T} = ntuple(i -> T(0.5), Val{N}())
    fx::T = one(T)
    logfx = zero(T)
    function g(zw, logfx, f, λ, EX)
        w, n = zw
        x = EX .+ n ./ √λ
        px = λ^T(1.5)*exp( w + λ*f( x ) - logfx )
        xpx = x .* px 
        px, xpx, x .* xpx
    end
#    init_reduce = init_reduce(T, Val{N}())
    while !converged
        let a1 = logfx, a2 = EX, a3 = λ
            fx, EX, VX = mapreduce(x -> g(x, a1, f, a3, a2), reduction_op, grid)::Tuple{T,NTuple{N,T},NTuple{N,T}}
        end#, init_reduce
        EX = EX ./ fx
        VX = @. sqrt( VX / fx - abs2(EX) )
        iterations += 1
        if maximum(VX) < ϵ
            converged = true
        elseif iterations == max_iter
            break
        end
        λ = step(λ, VX)
        logfx += log(fx)
        if verbose == true
            println("Iteration: $iterations\nEX: $EX, VX: $VX, λ: $λ.\n")
        end

    end
    PincusResult(EX, VX, converged, iterations)
end

function maximize(f, grid::CLArray{Tuple{T,NTuple{N,T}},1}, ::Val{verbose} = Val{true}(), λ = T(0.1), max_iter::Integer = 30, ϵ = 1e-3, step = naive_step) where {N,T, verbose}
    iterations = 0
    converged = false
    EX::NTuple{N,T} = zero_tuple(T, Val{N}())
    VX::NTuple{N,T} = ntuple(i -> T(0.5), Val{N}())
    fx::T = one(T)
    logfx = zero(T)
    function g(zw, logfx, f, λ, EX)
        w, n = zw
        x = EX .+ n ./ √λ
        px = λ^T(1.5)*exp( w + λ*f( x ) - logfx )
        xpx = x .* px 
        px, xpx, x .* xpx
    end
    init = init_reduce(T, Val{N}())
    while !converged
        let a1 = logfx, a2 = EX, a3 = λ
            fx, EX, VX = mapreduce(x -> g(x, a1, f, a3, a2), reduction_op, init, grid)::Tuple{T,NTuple{N,T},NTuple{N,T}}
        end#, init_reduce
        EX = EX ./ fx
        VX = @. sqrt( VX / fx - abs2(EX) )
        iterations += 1
        if maximum(VX) < ϵ
            converged = true
        elseif iterations == max_iter
            break
        end
        λ = step(λ, VX)
        logfx += log(fx)
        if verbose == true
            println("Iteration: $iterations\nEX: $EX, VX: $VX, λ: $λ.\n")
        end

    end
    PincusResult(EX, VX, converged, iterations)
end
function threaded_maximize(f, grid::AbstractVector{Tuple{T,NTuple{N,T}}}, ::Val{verbose} = Val{true}(), λ = T(0.1), max_iter::Integer = 30, ϵ = 1e-3, step = naive_step) where {N,T, verbose}
    iterations = 0
    converged = false
    EX::NTuple{N,T} = zero_tuple(T, Val{N}())
    VX::NTuple{N,T} = ntuple(i -> T(0.5), Val{N}())
    fx::T = one(T)
    logfx = zero(T)
    function g(zw, logfx, f, λ, EX)
        w, n = zw
        x = EX .+ n ./ √λ
        px = λ^T(1.5)*exp( w + λ*f( x ) - logfx )
        xpx = x .* px 
        px, xpx, x .* xpx
    end
    nthread = Threads.nthreads()
    grid_partition = partition(grid, nthread)
    buffer = Vector{Tuple{T,NTuple{N,T},NTuple{N,T}}}(nthread)
    while !converged
        let a1 = logfx, a2 = EX, a3 = λ
            fx, EX, VX = threaded_mapreduce!(buffer, x -> g(x, a1, f, a3, a2), reduction_op, grid_partition)::Tuple{T,NTuple{N,T},NTuple{N,T}}
        end
        EX = EX ./ fx
        VX = @. sqrt( VX / fx - abs2(EX) )
        iterations += 1
        if maximum(VX) < ϵ
            converged = true
        elseif iterations == max_iter
            break
        end
        λ = step(λ, VX)
        logfx += log(fx)
        if verbose == true
            println("Iteration: $iterations\nEX: $EX, VX: $VX, λ: $λ.\n")
        end

    end
    PincusResult(EX, VX, converged, iterations)
end

logit(x) = log( x/(one(x)-x) )
inv_logit(x) = one(x) / (one(x) + exp(-x))
function uniboxmaximize(f, grid::AbstractVector{NTuple{2,T}}, verbose::Val{v} = Val{false}(), EX::T = T(0.5), λ::T = T(0.1), max_iter::Int = 50, ϵ = 1e-6, step = naive_step) where {T,v}
    iterations = 0
    converged = false
    EX = logit(EX)
    VX = one(T)
    fx = one(T)
    logfx = zero(T)
    function g(zw, logfx, f, λ, EX)
        n, w = zw
        x = EX + n/√λ
        px = λ^T(1.5)*exp( w + λ*f(inv_logit(x)) - logfx )
        xpx = x * px 
        px, xpx, x*xpx
    end
    func = f
    while !converged
        let a1 = logfx, a2 = λ, a3 = EX
            fx, EX, VX = mapreduce(x -> g(x, a1, f, a2, a3), sum_tup, grid )
        end
        EX = EX / fx
        VX = VX / fx - abs2(EX)
        iterations += 1
        if VX < ϵ
            converged = true
        elseif iterations == max_iter
            break
        end
        if v == true
            println("Iteration: $iterations\nEX: $(inv_logit(EX))\nVX: $VX\nλ: $λ.\n")
        end
        λ = step(λ, VX)
        logfx += log(fx)
    end
    PincusResult((inv_logit(EX),), (VX,), converged, iterations)
end

function gauss_hermite_tup(N::Int, ::Type{T} = Float64) where T
    n, w = gausshermite(N)
    [((√T(2)*T(n[i])), log( T(w[i])/T(√π) ) + abs2(T(n[i]))/2 ) for i ∈ 1:N]
end
function gauss_hermite_tup2(N::Int, ::Type{T} = Float64) where T
    n, w = gausshermite(N)
    [((√T(2)*n[i]), T(w[i])/√π) for i ∈ 1:N]
end
function partition(x, n)
    N = length(x)
    [x[1+N*(i-1)÷n:N*i÷n] for i ∈ 1:n]
end
function partition(x::A, ::Val{n}) where {A,n}
    N = length(x)
    ntuple(i -> x[1+N*(i-1)÷n:N*i÷n], Val{n}())
end

function threaded_mapreduce!(buffer, f, op, vec)
#    tid = Vector{Int}(length(vec))
    Threads.@threads for i in eachindex(vec)
        buffer[i] = mapreduce(f, op, vec[i])
#        tid[i] = Threads.threadid()
    end
#    println("Thread ids:", tid)
    reduce(op, buffer)
end
function threaded_uniboxmaximize(f, grid::AbstractVector{NTuple{2,T}}, verbose::Val{v} = Val{false}(), EX::T = T(0.5), λ::T = T(0.1), max_iter::Integer = 30, ϵ = 1e-6, step = naive_step) where {T,v}
    iterations = 0
    converged = false
    scale = T(0.5)
    EX = logit(EX)
    VX = one(T)
    fx = one(T)
    logfx = zero(T)
    function g(zw, logfx, f, λ, EX, scale)
        n, w = zw
        x = n*scale + EX
        px = w*exp( λ*f( inv_logit(x) )+abs2(n/Cuint(2)) - logfx )
        xpx = x * px
        px, xpx, x * xpx
    end
    init_reduce = (zero(T),zero(T),zero(T))
    #nthread = Threads.nthreads()
    grid_partition = partition(grid, Val{16}())::NTuple{16,Vector{Tuple{BigFloat,BigFloat}}}
    buffer = Vector{typeof(init_reduce)}(16)
    while !converged
        let a1 = logfx, a2 = λ, a3 = EX, a4 = scale
            fx, EX, VX = threaded_mapreduce!(buffer, x -> g(x, a1, f, a2, a3, a4), sum_tup, init_reduce, grid_partition )
        end
        EX = EX / fx
        VX = VX / fx - abs2(EX)
        scale = √VX
        iterations += 1
        if VX < ϵ
            converged = true
        elseif iterations == max_iter
            break
        end
        λ = step(λ, VX)
        if v == true
            println("Iteration: $iterations\nEX: $inv_logit(EX), VX: $VX, λ: $λ.\n")
        end
        logfx += log(fx)
    end
    PincusResult((inv_logit(EX),), (VX,), converged, iterations)
end


function unimaximize(f, grid::AbstractVector{NTuple{2,T}}, λ = T(0.1), max_iter::Integer = 100, ϵ = 1e-3, step = naive_step, ::Val{verbose} = Val{false}()) where {T,verbose}
    iterations = 0
    converged = false
    EX = zero(T)
    VX = one(T)
    logfx = zero(T)
    function g(nw, logfx, EX, VX, f, λ)
        n, w = nw
        fx = w*exp(λ * f( VX * n + EX ) - logfx )
        nfx = n * fx
        fx, nfx, n * nfx
    end
    while !converged
        fx, EX, VX = mapreduce(x -> g(x, EX, VX, f, λ), sum_tup, (zero(T),zero(T),zero(T)), grid )
        EX = EX / fx
        VX = √( VX / fx - abs2(EX) )
        iterations += 1
        if VX < ϵ
            converged = true
        elseif iterations == max_iter
            break
        end
        λ = step(λ, VX)
        logfx += log(fx)
        if verbose == true
            println("Iteration: $iterations\nEX: $EX, VX: $VX, λ: $λ.\n")
        end
    end
    PincusResult((EX,), (VX,), converged, iterations)
end

naive_step(λ::T, VX) where T = λ*T(1.5)
cautious_naive_step(λ::T, VX) where T = λ*T(1.1)