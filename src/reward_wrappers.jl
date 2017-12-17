# --------------------------------
# Reward Modifying Wrappers
# --------------------------------

abstract type AbstractRewardWrapper <: AbstractGymWrapper end
macro reward_fields()
    quote
        wrp_reward::Float64 # reward given by this wrapper at prev `step!`
        wrp_total_reward::Float64 # sum of rewards for the entire ep given only by this wrapper
    end |> esc
end

const reward_defaults = (0.0, 0.0)

"""
`step_reward(wrpenv::AbstractRewardWrapper, r::Real)`
Called at each `step!` to allow modifying the reward for the step
Overload this to specify custom handling of rewards at each `step!`, should
return only the reward that this wrapper applies
"""
function step_reward(wrpenv::AbstractRewardWrapper, r::Real)
    error("step_reward(wrpenv, $r) not implemented for wrpenv of type ", typeof(wrpenv))
end

function reward(wrpenv::AbstractRewardWrapper)
    wrp_reward(wrpenv) + reward(wrpenv.env)
end

function total_reward(wrpenv::AbstractRewardWrapper)
    wrp_total_reward(wrpenv) + total_reward(wrpenv.env)
end

wrp_reward(wrpenv::AbstractRewardWrapper)       = wrpenv.wrp_reward
wrp_total_reward(wrpenv::AbstractRewardWrapper) = wrpenv.wrp_total_reward

function Reinforce.step!(wrpenv::AbstractRewardWrapper, args...)
    r, s = Reinforce.step!(wrpenv.env, args...)
    wrp_reward = step_reward(wrpenv, r) # subtypes implement
    wrpenv.wrp_reward = wrp_reward
    wrpenv.wrp_total_reward += wrp_reward
    wrp_reward, s
end

function Reinforce.reset!(wrpenv::AbstractRewardWrapper)
    wrpenv.wrp_reward = wrpenv.wrp_total_reward = 0.0
    Reinforce.reset!(wrpenv.env)
end

"""
The total bonus/penalty on rewards this wrapper has applied for the current episode
"""
function wrp_total_reward(::Type{T}, env) where {T <: AbstractRewardWrapper}
    wrpenv = getwrapper(T, env)
    wrp_total_reward(wrpenv)
end

survival_reward(env) = wrp_total_reward(SurvivalRewardWrapper, env)
actentropy_reward(env)  = wrp_total_reward(ActionEntropyWrapper, env)
rage_reward(env)     = wrp_total_reward(RageQuitWrapper, env)

gym_reward(env) = total_reward(gymenv(env))

# --------------------------------
# SurvivalRewardWrapper()
# --------------------------------
"""
```
SurvivalRewardWrapper(env::AbstractGymEnv, reward_thresh::Float64,
                      reward_cap::Float64, min_steps::Int, rstep::Float64)
```
if `total_reward(env)` is less than `reward_thresh`, adds a survival reward of
`rstep` to the reward of the wrapped enviornment at each timestep

The total_reward given in an episode will be between 0.0 and `reward_cap` and
can be accessed with `survival_reward(env)`

`reward_cap` should prob be smaller than the smallest reward an agent can get,
to avoid over encouraging staying alive vs winning.
"""
mutable struct SurvivalRewardWrapper <: AbstractRewardWrapper
    env::AbstractGymEnv
    reward_thresh::Float64
    reward_cap::Float64
    min_steps::Int
    max_steps::Int
    rstep::Float64 # reward per step
    @reward_fields()
end

function SurvivalRewardWrapper(env::AbstractGymEnv, reward_thresh::Float64,
                               reward_cap::Float64, min_steps::Int, rstep::Float64)
    max_steps = floor(Int, 1.0/rstep) + min_steps
    SurvivalRewardWrapper(env, reward_thresh, reward_cap, min_steps, max_steps,
                          rstep, reward_defaults...)
end

"""
SurvivalRewardWrapper(env::AbstractGymEnv, reward_thresh::Float64,
                      reward_cap::Float64, min_steps::Int, max_steps::Int)

Convenience constructor to set `rstep` (reward per step) as `1.0/(max_steps - min_steps)`
"""
function SurvivalRewardWrapper(env::AbstractGymEnv, reward_thresh::Float64,
                               reward_cap::Float64, min_steps::Int, max_steps::Int)
    rstep = 1.0/(max_steps - min_steps)
    SurvivalRewardWrapper(env, reward_thresh, reward_cap, min_steps, max_steps,
                          rstep, reward_defaults...)
end

function step_reward(wrpenv::SurvivalRewardWrapper, r::Real)
    rstep = 0.0
    if total_reward(wrpenv.env) < wrpenv.reward_thresh &&
       wrpenv.min_steps < elapsed_steps(wrpenv) <= wrpenv.max_steps
           rstep = wrpenv.rstep
    end
    rstep
end

elapsed_steps(wrpenv::AbstractGymWrapper) = gymenv(wrpenv).pyenv[:_elapsed_steps]
elapsed_steps(env::GymEnv) = env.pyenv[:_elapsed_steps]

# --------------------------------
# ActionEntropyWrapper
# --------------------------------
"""
Gives reward for diversity of actions. Variety is the spice of (artificial)
life. The Action entropy wrapper isn't implemented like a standard
AbstractRewardWrapper in that it only changes the total_reward for an episode,
not the reward for each step
"""
mutable struct ActionEntropyWrapper{T} <: AbstractRewardWrapper
    env::AbstractGymEnv
    reward_thresh::Float64
    reward_max::Float64
    min_acceptable_ent_ratio::Float64
    ignore_actions::Vector{<:Any}
    num_actions::Int
    max_ent::Float64
    ep_actions::Vector{T}
    @reward_fields()
end

function ActionEntropyWrapper(
        env::AbstractGymEnv, reward_thresh::Float64, reward_max::Float64,
        min_acceptable_ent_ratio::Float64, ignore_actions::Vector{<:Any},
        actionT::Type=Int
    )
    num_actions = actions(env) |> length
    ActionEntropyWrapper(env, reward_thresh, reward_max, min_acceptable_ent_ratio,
                         ignore_actions, num_actions, -log(1/num_actions),
                         actionT[], reward_defaults...)
end

"""
Get reward between 0.0 and wrpenv.reward_max for mixing it up
"""
function total_reward(wrpenv::ActionEntropyWrapper)
    base_reward = total_reward(wrpenv.env)
    entr_bonus = 0.0
    base_reward < wrpenv.reward_thresh &&
        (entr_bonus = entropy_reward(wrpenv))
    monotony_penalty = 1.0
    if base_reward >= wrpenv.reward_thresh &&
      entropy_ratio(wrpenv) < wrpenv.min_acceptable_ent_ratio
        monotony_penalty = entropy_ratio(wrpenv)
    end
    monotony_penalty * (base_reward + entr_bonus)
end

# Override wrp_total_reward for action entropy since unlike other wrappers, it's
# a non-seperable function of total_reward(wrpenv.env), and so is not accumulated
# in wrpenv.wrp_total_reward at each step
wrp_total_reward(wrpenv::ActionEntropyWrapper) =
    total_reward(wrpenv) - total_reward(wrpenv.env)

"""
More diverse actions => higher entropy => entropy_ratio closer to 1.0
Less diverse actions => lower entropy  => entropy_ratio closer to 0.0
"""
function entropy_ratio(wrpenv::ActionEntropyWrapper)
    entr = proportionmap(wrpenv.ep_actions) |> values |> collect |> entropy
    entr/wrpenv.max_ent # ∈ [0,1]
end

function entropy_reward(wrpenv::ActionEntropyWrapper)
    wrpenv.reward_max * entropy_ratio(wrpenv)
end

function Reinforce.step!(wrpenv::ActionEntropyWrapper, s, a)
    r, s′ = Reinforce.step!(wrpenv.env, s, a)
    !(a in wrpenv.ignore_actions) && push!(wrpenv.ep_actions, a)
    r, s′
end

Reinforce.step!(wrpenv::ActionEntropyWrapper, a) =
    Reinforce.step!(wrpenv::ActionEntropyWrapper, state(wrpenv), a)

function Reinforce.reset!(wrpenv::ActionEntropyWrapper)
    empty!(wrpenv.ep_actions)
    Reinforce.reset!(wrpenv.env)
end

action_entropy_env(wrpenv::ActionEntropyWrapper) = wrpenv
action_entropy_env(wrpenv::AbstractGymWrapper) = action_entropy_env(wrpenv.env)
action_entropy_env(env::OpenAIGym.GymEnv) = env
