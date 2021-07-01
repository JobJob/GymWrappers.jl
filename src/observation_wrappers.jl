import Images: imresize, normedview
import ImageTransformations: imresize!
export GreyMeanWrapper, GreyChanWrapper, DownsizeWrapper, MultiFrameWrapper, state_img
using PyCall

# --------------------------------
# Greyscale Wrappers
# --------------------------------
mat_like_state(::Type{T}, env) where T = Matrix{T}(size(state(env))[1:2])

abstract type GreyscaleGymWrapper <: AbstractGymWrapper end

"""
`GreyMeanWrapper(env::AbstractGymEnv, ::Type{T} = UInt8, zero_centre::Bool = false)`

For each pixel in `state(env)`, assumed to be encoded as RGB UInt8s in [0,255].
Average the R, G, and B channels, and normalise so that
`state(GreyMeanWrapper(env))` is a Matrix with all values in the interval [0,1]

If `zero_centre` is true subtract 0.5 from each channel, so that all values lie
in the interval [-0.5, 0.5]. Accordingly, `zero_centre` is not supported for
types that don't support negative values, e.g. `UInt8` and Images.jl's
FixedPoint types.
"""
struct GreyMeanWrapper{T} <: GreyscaleGymWrapper
    env::AbstractGymEnv
    zero_centre::Bool
    img::Matrix{T} # initialise on creation, update each `step!`
end

function GreyMeanWrapper(env::AbstractGymEnv, ::Type{T} = UInt8,
  zero_centre::Bool = false) where T
    img = mat_like_state(T, env)
    rgb2greymean!(img, state(env), zero_centre)
    GreyMeanWrapper{T}(env, zero_centre, img)
end

# @generated function rgb2greymean!(wrap_img::Matrix{FPT}, state_img::Array{UT,3}, zero_centre::Bool) where
#   {UT <: Unsigned, FPT <: Normed}
#     @assert sizeof(UT) == sizeof(FPT)
#     # must divide each UInt8 by 3 rather than their sum, to avoid overflow
#     quote
#         wrap_img .= reinterpret.($FPT,
#                       @view(state_img[:,:,1]) .÷ 0x3 .+
#                       @view(state_img[:,:,2]) .÷ 0x3 .+
#                       @view(state_img[:,:,3]) .÷ 0x3
#                     )
#     end
# end

# fast conversion of PyArray{UInt8} to Array{N0f8}
# @benchmark unsafe_wrap(Array, convert(Ptr{N0f8}, pyimg.data), size(pyimg))

"""
Returned array's eltype == input array's eltype == UInt8
"""
function rgb2greymean!(wrap_img::Matrix{UInt8}, state_img::AbstractArray{UInt8,3},
    zero_centre::Bool)
    # must divide each UInt8 by 3 rather than their sum, to avoid overflow
    wrap_img .= (@view(state_img[:,:,1]) .÷ 0x3 .+
                 @view(state_img[:,:,2]) .÷ 0x3 .+
                 @view(state_img[:,:,3]) .÷ 0x3)
end

"""
Returned array's eltype (TR) != input array's eltype (TI)
and TR <: AbstractFloat and TI <: Integer (pixel vals ∈ [0,255]),
so we scale by 255.0 so output pixels are ∈ [0, 1.0]
"""
function rgb2greymean!(wrap_img::Matrix{TR}, state_img::AbstractArray{TI,3},
    zero_centre::Bool) where {TR <: AbstractFloat, TI <: Integer}
    centre_const = zero_centre ? TR(0.5) : TR(0.0)
    wrap_img .= (TR.(@view(state_img[:,:,1])) .+
                 TR.(@view(state_img[:,:,2])) .+
                 TR.(@view(state_img[:,:,3]))) ./ TR(255.0*3.0)
end

function Reinforce.step!(wrpenv::GreyMeanWrapper, args...)
    r, s = Reinforce.step!(wrpenv.env, args...)
    rgb2greymean!(wrpenv.img, state(wrpenv.env), wrpenv.zero_centre)
    r, state(wrpenv)
end

function Reinforce.state(wrpenv::GreyMeanWrapper)
    wrpenv.img
end

"""
`GreyChanWrapper=>(channel::Int, )`
Turn the rgb to a grey image by selecting just one Channel, and subtracting
128 from all pixels.

state(env) assumed to be encoded as RGB UInt8s in [0,255].
"""
struct GreyChanWrapper{T} <: GreyscaleGymWrapper
    env::AbstractGymEnv
    chan::Int
    zero_centre::Bool
    img::Matrix{T} # allocate on creation, update each `step!`
end

function GreyChanWrapper(env::AbstractGymEnv, chan = 1, zero_centre::Bool = false)
    img = mat_like_state(env)
    rgb2greychan!(img, state(env), chan, zero_centre)
    GreyChanWrapper(env, chan, zero_centre, img)
end

function rgb2greychan!(wrap_img, state_img, chan::Int, zero_centre::Bool)
    zero_factor = zero_centre ? 128.0 : 0.0
    wrap_img .= ((@view state_img[:, :, chan]) .- zero_factor) ./ 255.0
end

function Reinforce.step!(wrpenv::GreyChanWrapper, args...)
    r, s = Reinforce.step!(wrpenv.env, args...)
    rgb2greychan!(wrpenv.img, state(wrpenv.env), wrpenv.chan, wrpenv.zero_centre)
    r, state(wrpenv)
end

function Reinforce.state(wrpenv::GreyChanWrapper)
    wrpenv.img
end

# --------------------------------
# Multi-Frame Wrapper
# --------------------------------
"""
MultiFrameWrapper=>(num_frames::Int)
"""
mutable struct MultiFrameWrapper{T,N} <: GreyscaleGymWrapper
    env::AbstractGymEnv
    nframes::Int
    frame_idx::Int # final index into all-frames, will be incremented on each
        # step, like a sliding window, to avoid copying
    recent_frames::Array{T, N} # allocate on creation, update each step
        # size(recent_frames) == (size(state(wrpenv.env))..., wrpenv.nframes)
end

function init_with_repeated_frame(env, nframes)
    res = cat(ndims(state(env))+1, (state(env) for _ in 1:nframes)...)
    # @dbg ndims(state(env))+1 size(res) nframes size(state(env))
    res
end

"""
`MultiFrameWrapper{T,N}(env, nframes::Int=2)`

N is the rank of the recent_frames tensor, so is one more than the rank of
state(env), i.e. ndims(state(env))+1, e.g.
Atari RGB: state(env) is (m, n, c) so is 3 + 1 = 4
Atari Grey: state(env) is (m, n)   so is 2 + 1 = 3
"""
MultiFrameWrapper(env::AbstractGymEnv, nframes = 2) =
    MultiFrameWrapper(env, nframes, 1, init_with_repeated_frame(env, nframes))

"""
`Reinforce.state{N}(wrpenv::MultiFrameWrapper{T, N})`

Return a view of the last `wrpenv.nframes` frames. Returns `wrpenv.nframes` of
`wrpenv.recent_frames` starting from wrpenv.recent_frames[:, :,( :,)
wrpenv.frame_idx] wrapping around the last dim as needed. The returned array is
of size `(m, n, c * wrpenv.nframes)`, where (m, n, c) are the width, height, and
number of channels of the wrapped env (i.e. of `state(wrpenv.env)`)

assumes N ∈ [3,4], i.e. state(wrpenv.env) is 2D (Greyscale), or 3D (RGB)
"""
function Reinforce.state{T, N}(wrpenv::MultiFrameWrapper{T, N})
    # assert(N ∈ (3,4))
    newsize = ones(Int, 3)
    newsize[1:N-1] .= size(wrpenv.recent_frames)[1:N-1]
    newsize[3] *= wrpenv.nframes # stack all channels from all frames along dim 3
    # @dbg size(wrpenv.recent_frames) newsize wrpenv.nframes N wrpenv.frame_idx
    res = reshape(recent_frames(wrpenv), newsize...)
    # @dbg "state" size(res) typeof(res)
    res
end

"""
`state_img{N}(wrpenv::MultiFrameWrapper{T, N})`
Get an image representation of the state for rendering
returns state(env) with frames stacked side by side
"""
function state_img{T, N}(wrpenv::MultiFrameWrapper{T, N})
    newsize = ones(Int, 3)
    newsize[1:N-1] .= size(wrpenv.recent_frames)[1:N-1]
    newsize[2] *= wrpenv.nframes # stack all images side by side
    newsize[3] == 1 && pop!(newsize) # squeeze for Greyscale
    # @dbg size(wrpenv.recent_frames) newsize wrpenv.nframes N wrpenv.frame_idx
    res = reshape(recent_frames(wrpenv), newsize...)
    # @dbg "state_img" size(res) typeof(res)
    res
end

function recent_frames{T, N}(wrpenv::MultiFrameWrapper{T, N})
    recent_window_indices =
        wraprangeidxs(wrpenv.frame_idx,
                      mod1(wrpenv.frame_idx - 1, wrpenv.nframes),
                      wrpenv.nframes)
    # @dbg N recent_window_indices
    view(wrpenv.recent_frames, colons(N-1)..., recent_window_indices)
end

"""
`state_img(env::AbstractEnvironment)`
Get an image representation of the state
returns state(env) by default
"""
state_img(env::AbstractEnvironment) = state(env)
state_img(wrpenv::AbstractGymWrapper) = state_img(wrpenv.env)

"""
performs:
wrpenv.recent_frames[:, :, ..., :, wrpenv.frame_idx] .= state(wrpenv.env)
and increments `wrpenv.frame_idx`, wrapping it back to 1 if it's `> wrpenv.nframes`
"""
function set_next_frame!{T, N}(wrpenv::MultiFrameWrapper{T, N}, state_img)
    env_state_ndims = N-1
    setindex!(wrpenv.recent_frames, state_img,
              colons(env_state_ndims)..., wrpenv.frame_idx)
    wrpenv.frame_idx = mod1(wrpenv.frame_idx + 1, wrpenv.nframes)
end

function Reinforce.step!(wrpenv::MultiFrameWrapper, args...)
    r, s = Reinforce.step!(wrpenv.env, args...)
    set_next_frame!(wrpenv, state(wrpenv.env))
    r, state(wrpenv)
end

# --------------------------------
# Downsize Wrapper
# --------------------------------
"""
DownsizeWrapper=>(newsize::Tuple{Int,Int})

Return an wrapped env such that all channels of state(env) are scaled to
`newsize`. The input env's state is assumed to be of size `(h, w, c)` where w,h
are the width and height of the image, and c is the number of channels).  The
wrapped state(env), will thus have size `newsize`

Assumes state(env) stays constant size.
"""
struct DownsizeWrapper{T, N} <: GreyscaleGymWrapper
    env::AbstractGymEnv
    scaled_img::Array{T, N} # allocate on creation, update each step
end

"""
`DownsizeWrapper(env, scaler::Tuple{Float64}=(0.5, 0.5))`
"""
function DownsizeWrapper(env, scaler = 0.5)
    newsize = size(state(env)) |> collect
    newsize[1:2] .= round.(newsize[1:2].*scaler)
    DownsizeWrapper(env, Tuple(newsize))
end

"""
`DownsizeWrapper(env, newsize::Tuple{Int,Int})`
"""
function DownsizeWrapper(env, newsize::Tuple{Int,Int})
    DownsizeWrapper(env, imresize(state(env), newsize))
end

"""
`DownsizeWrapper(env, newsize::Int...)`
"""
DownsizeWrapper(env, newsize::Int...) = DownsizeWrapper(env, newsize)

"""
`Reinforce.state(wrpenv::DownsizeWrapper)`

see `DownsizeWrapper(env, scaler)`.
"""
function Reinforce.state(wrpenv::DownsizeWrapper)
    wrpenv.scaled_img
end

function Reinforce.step!(wrpenv::DownsizeWrapper, args...)
    r, s = Reinforce.step!(wrpenv.env, args...)
    imresize!(wrpenv.scaled_img, state(wrpenv.env))
    r, state(wrpenv)
end

downsized(wrapped_env::AbstractGymWrapper) = downsized(wrapped_env.env)
downsized(wrapped_env::DownsizeWrapper) = wrapped_env
downsized(env::GymEnv) = error("env $env doesn't have a DownsizeWrapper")
