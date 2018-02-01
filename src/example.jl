

f(x::Float64) = cos(40π*x) + (1 - (2π*x-π)^2)
f(x) = cos( Cuint(40)π*x ) + (Cuint(1) - abs2(Cuint(2)π*x-π))


#Here I want everything to be immutable, stack allocated isbits.
#This limits the size of the problem.
struct TData{N,P,T}
    x::NTuple{N,{NTuple{P,T}}}
    ν::T
end

@generated triangle(::Val{N}) where N = Val{N * (N+1) ÷ 2}()
@generated dim_of_triangle(::Val{N}) where N = Val{(isqrt(1+8N)-1)÷2}()
@generated function decompose_param(::Val{N}) where N
    P = (isqrt(9+8N)-3)÷2
    vP = Val{P}()
    Cuint(P), vP, triangle(vP)
end
function split_params(x::NTuple{N,T}) where {N,T}
    P, vP, vT = decompose_param(Val{N}())
    (ntuple(i -> x[i], vP),
    ntuple(i -> x[i+P], vT))
end

like(::Type{Integer}, ::Type{Float64}) = Int
like(::Type{Integer}, ::Type{Float32}) = Cuint

@generated function ℓ(θ::NTuple{D,T}, t::TData{N,P,T}) where {D,N,P,T}
    I = like(Integer, T)
    quote
        log_density = zero(T)
        #Summing logs is slower but more accurate
        #than taking the log of products
        @nexprs $P i -> begin 
            log_density += log(θ[I(($P+1)*i - (i-1)*i÷2)])
        end
        log_density *= N
        @nexprs $N i -> begin
            ##calculate quadform
            @nextract 4 q k -> zero(T)
            @nexprs $P j -> begin
                δ = θ[I(j)] - t.x[I(i)][I(j)]
                q_j += δ * exp(θ[I(j*(P+1)-((j-1)*j)÷2)])#The diagonal elements.
                @nexprs $P-j k -> begin
                    q_{k+j} += δ * θ[I(k + j*(P+1)-((j-1)*j)÷2)]
                end
            end
            @nexprs $P-1 j -> begin
                q_1 += q_{j+1}
            end
            log_density -= (t.ν + $P)/$I(2)*($I(1)+q_1/t.ν)
        end
        log_density
    end
end

struct UpperTriangle{T, A <: AbstractArray{T}, P} <: AbstractArray{T,2}
    vals::A
end
@inline function UpperTriangule(x::NTuple{P,T}) where {P,T}
    UpperTriangle{T,Array{T,1},dim_of_triangle(Val{P}())}(Array(x))
end
Base.size(::UpperTriangle{T,A,P}) where {T,A,P} = (P,P)
@generated Base.length(::UpperTriangle{T,A,P}) where {T,A,P} = P*(P+1)÷2
Base.getindex(U::UpperTriangle, i) = U.vals[i]
@inline function Base.getindex(U::UpperTriangle{T,A,P}, i, j) where {T,A,P}
    U.vals[j + (i-1)P - (i-1)i÷2]
end
@inline function Base.setindex!(U::UpperTriangle, v, i)
    U.vals[i] = v
end
@inline function Base.setindex!(U::UpperTriangle{T,A,P}, v, i, j) where {T,A,P}
    U.vals[j + (i-1)P - (i-1)i÷2] = v
end
#Lets not bother implementing any more of our own types.
function crossprod(U::UpperTriangle{T,A,P}) where {T,A,P}
    Σ = fill(zero(T),P,P)
    #It's late. Σ is column major, and I really ought to have implemented U that way. Ugh.
    #I think I chose row major because it made something in
    #the quadratic form a little more convenient.
    #Maybe it would be more efficient to actually transpose?
    for i ∈ 1:P
        for j ∈ 1:i
            for k ∈ 1:j
                Σ[j,i] += U[k,i]*U[k,j]
            end
        end
    end
    Symmetric(Σ)
end


#Lazy implementation.
#I should be using linear indices, not Cartesian.
#This implementation is row (NOT COLUMN) major
#so at least the memory access pattern should be more or less correct.
#Alternatively, a generated function with for loops -> @nexpr would let all the indices be determined at compile time.
#LLVM may take care of all this; if it does, it'd do it better.
#Benchmark.
function inv!(U::UpperTriangular{T,A,P}) where {T,A,P}
    for i ∈ 1:P
        Uᵢᵢ = U[i,i]
        for j ∈ i+1:P
            Uᵢⱼ = U[i,j] * Uᵢᵢ
            for k ∈ i+1:j-1
                Uᵢⱼ += U[k,j] * U[i,k]
            end
            U[i,j] = - Uᵢⱼ / U[j,j]
        end
    end
    U
end

function regenerate_t_params(θ::NTuple{D,T}) where {D,T}
    P, vP, vT = decompose_param(Val{D}())
    μ = [θ[i] for i ∈ 1:P]
    
    U = UpperTriangle{T,Array{T,1},dim_of_triangle(Val{P}())}(θ[P+1:D])
    for i ∈ 1:P
        U[i,i] = exp(U[i,i])
    end
    inv!(U)
    μ, crossprod(U)
end