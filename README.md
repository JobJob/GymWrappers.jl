## Wrappers for OpenAIGym Environments

Useful for reproducing Reinforcement Learning results using Atari envs.

Some of the wrappers were adapted from https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

### Install
```
Pkg.clone("https://github.com/JobJob/GymWrappers.jl")
```
might be needed
```
Pkg.rm("OpenAIGym")
Pkg.clone("https://github.com/JobJob/OpenAIGym.jl")
    Pkg.checkout("OpenAIGym", "total-reward")
Pkg.rm("Reinforce")
Pkg.clone("https://github.com/JobJob/Reinforce.jl")
    Pkg.checkout("Reinforce", "total-reward")
```

### Basic Usage
```
using OpenAIGym, GymWrappers
env = maybe_wrap(GymEnv("BreakoutNoFrameskip-v4"), wrap_deepmind=true))
```
Will wrap the env so it's essentially like Deepmind's work in
[Human-level control through deep reinforcement learning (Mnih et al. 2015)](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/)

Access the unwrapped env with
```
gymenv(env) # Julia wrapper
gymenv(env).pyenv # python env (usually wrapped with TimeLimit etc.)
gymenv(env).pyenv[:unwrapped] # fully unwrapped env
```

Access a particular wrapper with, e.g.:
```
getwrapper(MultiFrameWrapper, env)
```

### Advanced usage
```
better_wrapper_specs = Dict{String, Vector{Pair{Type{T} where T <: AbstractGymWrapper, Tuple}}}(
    "Pong"=>[
        ActionSetWrapper=>([0, 2, 3],),
        RageQuitWrapper=>(7.0,),
    ],
    "Breakout"=>[
        ActionSetWrapper=>([0, 2, 3],),
    ],
    "deepmind_defaults"=>[
        # wrappers applied from inner most (top) -> outer most (bottom)
        EpisodicLifeWrapper=>(),
        GreyChanWrapper=>(2,),
        DownsizeWrapper=>((0.4, 0.525),), # (210, 160) -> (84, 84)
        MaxAndSkipWrapper=>(4,), # repeat action for 4 frames
        MultiFrameWrapper=>(4,), # stack last 4 frames
    ]
)

env = maybe_wrap(GymEnv("PongNoFrameskip-v4"),
                 wrapper_specs=better_wrapper_specs,
                 use_atari_defaults=true)
```
