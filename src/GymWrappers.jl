module GymWrappers

using OpenAIGym
import OpenAIGym: AbstractGymEnv, actionset
import Reinforce: reward, total_reward

abstract type AbstractGymWrapper <: AbstractGymEnv end

export AbstractGymEnv, AbstractGymWrapper, gymenv, getwrapper, maybe_wrap
# other exports in included files

# Defaults
for fn in [:state, :reset!, :finished, :actions, :step!, :reward, :total_reward]
    @eval Reinforce.$fn(wrapped_env::AbstractGymWrapper, args...; kwargs...) =
            Reinforce.$fn(wrapped_env.env, args...; kwargs...)
end

# for fn in [:render]
#     @eval OpenAIGym.$fn(wrapped_env::AbstractGymWrapper; kwargs...) =
#             OpenAIGym.$fn(wrapped_env.env; kwargs...)
# end

# With multiple wraps, eventually this will get to the actual GymEnv
getwrapper{WrapType <: AbstractGymEnv}(::Type{WrapType}, wrpenv::AbstractGymWrapper) =
    getwrapper(WrapType, wrpenv.env) # recurses down the wrappers

# If the env is of the right type, return it
getwrapper{WrapType <: AbstractGymWrapper}(::Type{WrapType}, env::WrapType) = env

# Special case when looking for a GymEnv, to win the method specialisation race
getwrapper(::Type{GymEnv}, env::GymEnv) = env

# If we're here we looked through all the wrappers and there weren't any `WrapType` wrappers
getwrapper{WrapType <: AbstractGymEnv}(::Type{WrapType}, env::GymEnv) =
    error("env is not wrapped with a wrapper of type $WrapType")

# short syntax for getting the gym env
gymenv(env::AbstractGymEnv) = getwrapper(GymEnv, env)

# The Business
"""
`maybe_wrap(env::AbstractGymEnv; wrapper_specs=default_wrapper_specs, wrap_deepmind=true)`

Wrap an environment using the wrapper_specs in `wrapper_specs[gamename]`, where
`gamename` is "Breakout" for `env.name == "BreakoutNoFrameskip-v4"`.

If `!haskey(wrapper_specs, gamename)` then the `env` is returned unwrapped.
Example wrapper spec:

```
default_wrapper_specs = Dict(
    "Pong"=>[ActionSetWrapper=>([2, 3],), GreyMeanWrapper=>(false,)]
)
```

Which will call `ActionSetWrapper(env, [2,3])`, `GreyMeanWrapper(env, false)`
"""
function maybe_wrap(env::AbstractGymEnv; wrapper_specs=default_wrapper_specs,
                    wrap_deepmind=true)
    gamename, gamever = env.name, env.ver
    replace(gamename, r"(NoFrameskip|Deterministic)","")
    wrapped_env = env
    all_wrappers = get(wrapper_specs, gamename, [])
    wrap_deepmind && append!(all_wrappers, wrapper_specs["deepmind_defaults"])
    for wrapper_spec in all_wrappers
        wrapper, args = wrapper_spec
        wrapped_env = wrapper(wrapped_env, args...)
    end
    wrapped_env
end

include("misc_utils.jl")
include("reward_wrappers.jl")
include("observation_wrappers.jl")
include("action_wrappers.jl")
include("episode_wrappers.jl")


# --------------------------------
# Default Wrappers
# --------------------------------
default_wrapper_specs = Dict{String, Vector{Pair{Type{T} where T <: AbstractGymWrapper, Tuple}}}(
    "deepmind_defaults"=>[
        # wrappers applied from inner most (top) -> outer most (bottom)
        EpisodicLifeWrapper=>(),
        GreyMeanWrapper=>(),
        # GreyChanWrapper=>(2,),
        DownsizeWrapper=>(84, 84), # (210, 160) -> (84, 84)
        MaxAndSkipWrapper=>(4,), # repeat action for 4 frames
        MultiFrameWrapper=>(4,), # stack last 4 frames
    ]
)

# --------------------------------
# Tests
# --------------------------------
#---
# envname = "Pong-v0"
# env = GymEnv(envname) |> maybe_wrap
# #---
# gryimg = OpenAIGym.render(env; mode="rgb_array")
# #---
# colorview(Gray, (env.env.env.state[:,:,3] ./ 255)) == gryimg
# #---
# imshow(gryimg)
# #---
# OpenAIGym.render(env) # should call imshow automatically
# #---
# actions(env)
#---
end
