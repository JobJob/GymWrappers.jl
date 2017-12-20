using IterTools
"""
`wraprange(start, stop, wrapat) = wraprange(start, 1, stop, wrapat)`
`wraprange(start, step, stop, wrapat)`

a range that wraps around back to 1, at `wrapat`.
Can be used for a sliding window index that wraps around when it reaches the size
of the dimension it's indexing

e.g.
```
julia> collect(wraprange(4,3,7))
7-element Array{Int64,1}:
 4
 5
 6
 7
 1
 2
 3

julia> collect(wraprange(4, 2, 3, 7))
4-element Array{Int64,1}:
 4
 6
 1
 3

julia> a = rand(2,5)
2×5 Array{Float64,2}:
 0.79923  0.280401  0.598733  0.213387  0.94785
 0.69775  0.246508  0.981085  0.318419  0.0183561

# return the view a[:,4:5] and a[:, 1:2]
julia> view(a, :, wraprangeidxs(4,2,5))
2×4 SubArray{Float64,2,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},Array{Int64,1}},false}:
 0.213387  0.94785    0.79923  0.280401
 0.318419  0.0183561  0.69775  0.246508
```
"""
wraprange(start, stop, wrapat) = wraprange(start, 1, stop, wrapat)
wraprange(start, step, stop, wrapat) = begin
    stop < start || return StepRange(start, step, stop)
    start <= wrapat ||
        throw(ArgumentError(
            "start ($start) must be smaller than wrapat ($wrapat)"))
    firstidx = step - (wrapat - start)%step
    chain(start:step:wrapat, firstidx:step:stop)
end

"""
`wraprangeidxs(args...)`

convenience for `collect(wraprange(args...))`
"""
wraprangeidxs(args...) = collect(wraprange(args...))

colons(n) = fill(Colon(), n)
