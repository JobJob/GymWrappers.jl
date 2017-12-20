export ActionSetWrapper, MaxAndSkipWrapper

# --------------------------------
# ActionSetWrapper
# --------------------------------
struct ActionSetWrapper <: AbstractGymWrapper
    env::AbstractGymEnv
    action_space::Vector{Int}
end
Reinforce.actions(wrpenv::ActionSetWrapper) = wrpenv.action_space
# TODO maybe put this into reinforce (generalise the GymEnv->AbstractEnvironment)
Reinforce.actions(env::GymEnv) = Reinforce.actions(env, env.state)

"""
MaxAndSkipWrapper(env::AbstractGymEnv, nskip=4)

Take the agent's chosen action over the next `nskip` steps. Return the observation
as the pixelwise max of the last two frames (the `nskip-1`th, and `nskip`th frame).

N.b. using `NoFrameskip` versions of gym environments is encouraged if using this
wrapper, as this wrapper does the skipping instead of the gym.
"""
#=
This wrapper is both an observation wrapper and an action wrapper, but we'll
put it in this file because the observation wrapper file is pretty long already
=#
mutable struct MaxAndSkipWrapper{T, N} <: AbstractGymWrapper
    env::AbstractGymEnv
    nskip::Int
    done::Bool
    recent_frame::Ref{Array{T, N}}
end

function MaxAndSkipWrapper(env::AbstractGymEnv, nskip = 4)
    s = state(env)
    MaxAndSkipWrapper{eltype(s), ndims(s)}(env, nskip, false, s)
end

function Reinforce.step!{T, N}(wrpenv::MaxAndSkipWrapper{T, N}, action)
    """Repeat action, sum reward, and max over last observations."""
    total_reward = 0.0
    done = false
    for i in 1:wrpenv.nskip
        reward, s = step!(wrpenv.env, action)
        done = Reinforce.finished(wrpenv.env)
        i == wrpenv.nskip - 1 && (wrpenv.recent_frame[] = s)
        total_reward += reward
        done && break
    end
    # Note that the observation on the done=True frame doesn't matter
    wrpenv.recent_frame[] = max.(wrpenv.recent_frame[], s)
    wrpenv.done = done
    return wrpenv.recent_frame[], total_reward
end

Reinforce.state{T, N}(wrpenv::MaxAndSkipWrapper{T, N}) = wrpenv.recent_frame[]

Reinforce.finished(wrpenv::MaxAndSkipWrapper) = wrpenv.done

function Reinforce.reset!(wrpenv::MaxAndSkipWrapper)
    wrpenv.done = false
    Reinforce.reset!(wrpenv.env)
    wrpenv.recent_frame[] = state(wrpenv.env)
end
