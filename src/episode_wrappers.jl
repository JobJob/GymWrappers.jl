export EpisodicLifeWrapper, RageQuitWrapper, PointLifeWrapper, PointlessLifeWrapper, FireOnReset, FireOnNewLife

# ################################
# Episodic Life Wrapper
# ################################
"""
`EpisodicLifeWrapper(env)`

Make end-of-life == end-of-episode, but only reset on true game over.
Done by DeepMind for the DQN and co. since it helps value estimation.

Adapted from https://github.com/openai/baselines/blob/param-noise-release/baselines/common/atari_wrappers.py#L50

\#Eplyf
"""
mutable struct EpisodicLifeWrapper <: AbstractGymWrapper
    env::AbstractGymEnv
    lives::Int
    lifelost::Bool
    was_real_done::Bool
end

function EpisodicLifeWrapper(env::AbstractGymEnv)
    # init was_real_done to true to ensure initial reset! is respected
    EpisodicLifeWrapper(env, ale_lives(env), false, true)
end

ale_lives(env::AbstractGymEnv) = gymenv(env).pyenv[:unwrapped][:ale][:lives]()

function Reinforce.step!(wrpenv::EpisodicLifeWrapper, s, a)
    r, s′ = Reinforce.step!(wrpenv.env, s, a)
    wrpenv.was_real_done = finished(wrpenv.env)
    # check current lives, make loss of life terminal,
    # then update lives to handle bonus lives
    lives = ale_lives(wrpenv)
    if lives < wrpenv.lives && lives > 0
        # for Qbert somtimes we stay in lives == 0 condtion for a few frames
        # so its important to keep lives > 0, so that we only reset once
        # the environment advertises done.
        wrpenv.lifelost = true
    end
    wrpenv.lives = lives
    return r, s′
end

function Reinforce.finished(wrpenv::EpisodicLifeWrapper, s)
    wrpenv.lifelost || wrpenv.was_real_done
end

"""
Reset only when lives are exhausted.
This way all states are still reachable even though lives are episodic,
and the learner need not know about any of this behind-the-scenes.
"""
function Reinforce.reset!(wrpenv::EpisodicLifeWrapper)
    # @dbg "reset! eplyf" wrpenv.was_real_done wrpenv.lifelost finished(wrpenv.env)
    if wrpenv.was_real_done
        s = Reinforce.reset!(wrpenv.env)
    else
        # no-op step to advance from terminal/lost life state
        r, s = Reinforce.step!(wrpenv.env, 0)
    end
    wrpenv.lifelost = false
    wrpenv.lives = ale_lives(wrpenv)
    wrpenv.was_real_done = finished(wrpenv.env)
    # @dbg "theend" finished(wrpenv.env) wrpenv.lives
    s
end


# ################################
# RageQuitWrapper
# ################################
"""
```
RageQuitWrapper(env::AbstractGymEnv, points_lost_limit::Int=1,
                point_lost_test=(env,r)->(r < 0))
```
Rage quit (end the episode immediately) if agent loses more than
`points_lost_limit` points. The function point_lost_test(env, r) is used to
determine if the point was lost, it defaults to just returning whether the
reward was less than 0.0
"""
mutable struct RageQuitWrapper <: AbstractGymWrapper
    env::AbstractGymEnv
    points_lost_limit::Int
    point_lost_test::Function
    points_lost::Int
end

function RageQuitWrapper(env::AbstractGymEnv,
                         points_lost_limit::Int=1,
                         point_lost_test=(env,r)->(r < 0))
    RageQuitWrapper(env, points_lost_limit, point_lost_test, 0)
end

function Reinforce.step!(wrpenv::RageQuitWrapper, s, a)
    r, s′ = Reinforce.step!(wrpenv.env, s, a)
    wrpenv.point_lost_test(wrpenv, r) && (wrpenv.points_lost += 1)
    r, s′
end

function Reinforce.finished(wrpenv::RageQuitWrapper, s)
    finished(wrpenv.env) || wrpenv.points_lost >= wrpenv.points_lost_limit
end

Reinforce.finished(wrpenv::RageQuitWrapper, args...) = wrpenv.ragequit

function Reinforce.reset!(wrpenv::RageQuitWrapper)
    wrpenv.points_lost = 0
    Reinforce.reset!(wrpenv.env)
end

#################################
# Point Life Wrapper
#################################
"""
`PointLifeWrapper(env, points_lost_limit, point_lost_test=(env, r)->(r <= 0.0))`

Alias for `RageQuitWrapper`
"""
const PointLifeWrapper = RageQuitWrapper

#################################
# Pointless Life Wrapper
#################################
"""
`PointlessLifeWrapper(env, points_lost_limit, point_lost_test=(env, r)->(r <= 0.0))`

Alias for `RageQuitWrapper`
"""
const PointlessLifeWrapper = RageQuitWrapper

"""
Press the fire button on reset for environments that are stuck until firing.
"""
struct FireOnReset <: AbstractGymWrapper
    env::AbstractGymEnv
    fire_action::Int
end

function FireOnReset(env::AbstractGymEnv, fire_action=1)
    action_meanings = gymenv(env).pygym[:unwrapped][:get_action_meanings]()
    @assert action_meanings[fire_action+1] == "FIRE" # 1-based so +1
    @assert length(action_meanings) >= 3
    FireOnReset(env, fire_action)
end

function reset!(wrpenv::FireOnReset)
    s = reset!(wrpenv.env)
    obs, r = Reinforce.step!(wrpenv.env, wrpenv.fire_action)
    done = Reinforce.finished(wrpenv.env)
    return obs
end

"""
Fire at the end of each life, required for some envs, e.g. breakout
when not wrapped in an EpisodicLifeWrapper
"""
mutable struct FireOnNewLife <: AbstractGymWrapper
    env::AbstractGymEnv
    fire_action::Int
    lives::Int
    openfire::Bool
end

function FireOnNewLife(env::AbstractGymEnv, fire_action = 1):
    @assert env.unwrapped.get_action_meanings()[fire_action+1] == "FIRE"
    @assert length(env.unwrapped.get_action_meanings()) >= 3
    FireOnNewLife(env, fire_action, 0, true)
end

function Reinforce.step!(wrpenv::FireOnNewLife, action)
    if wrpenv.openfire
        Reinforce.step!(wrpenv.env, wrpenv.fire_action)
        wrpenv.openfire = False
    end
    s, reward = Reinforce.step!(wrpenv.env, action)

    # check current lives, lose life => openfire on next step!,
    # then update lives to handle bonus lives
    lives = ale_lives(wrpenv)
    lives < wrpenv.lives && (wrpenv.openfire = True)
    wrpenv.lives = lives
    s, reward
end
