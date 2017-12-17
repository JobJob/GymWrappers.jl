## Wrappers for OpenAIGym Environments

Useful for reproduction of Reinforcement Learning results in Atari, and research in general.

Some of the wrappers were adapted from https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

### Install
```
Pkg.clone("https://github.com/JobJob/GymWrappers.jl")
```

Basic Usage:
```
using OpenAIGym, GymWrappers
env = maybe_wrap(GymEnv("BreakoutNoFrameskip-v4"), wrap_deepmind=true"))
```
Will wrap the env so it's essentially like Deepmind's work in
[Human-level control through deep reinforcement learning (Mnih et al. 2015)](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/)

Access the unwrapped env with
```
gymenv(env) # Julia wrapper
gymenv(env).pyenv # python version
gymenv(env).pyenv[:unwrapped] # python fully unwrapped version
```

Access a particular wrapper with, e.g.:
```
getwrapper(MultiFrameWrapper, env)
```

Advanced usage:
```
better_wrapper_specs = Dict{String, Vector{Pair{Type{T} where T <: AbstractGymWrapper, Tuple}}}(
    "Pong"=>[
        ActionSetWrapper=>([0, 2, 3],),
        RageQuitWrapper=>(7.0,),
                   # reward_thresh, reward_max, min_acceptable_ent_ratio, ignore_actions
        ActionEntropyWrapper=>(1.0, 0.5, 0.2, []),
        SurvivalRewardWrapper=>(2.0, 1.0, 99, 500)
    ],
    "Breakout"=>[
        ActionSetWrapper=>([0, 2, 3],),
                   # reward_thresh, reward_max, min_acceptable_ent_ratio, ignore_actions
        ActionEntropyWrapper=>(1.0, 0.5, 0.2, [1]),
    ],
    "deepmind_defaults"=>[
        EpisodicLifeWrapper=>(),
        # GreyChanWrapper=>(2,),
        DownsizeWrapper=>((0.4, 0.525),), # (210,160) -> 84x84 like in deepmind
        MaxAndSkipWrapper=>(4), # skip 4 frames
        MultiFrameWrapper=>(4,),
    ]
)

env = maybe_wrap(make_env("PongNoFrameskip-v4"),  wrapper_specs=better_wrapper_specs,
                    use_atari_defaults=true)")
```
