default_wrapper_specs = Dict{String, Vector{Pair{Type{T} where T <: AbstractGymWrapper, Tuple}}}(
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
        RageQuitWrapper=>(7,),
        GreyMeanWrapper=>(),
        # GreyChanWrapper=>(2,),
        DownsizeWrapper=>(84, 84)
        MultiFrameWrapper=>(4,), # stack last 4 frames
        MaxAndSkipWrapper=>(4,), # repeat action for 4 frames
    ]
)
