from gym.envs.registration import register

register(
    id='reacher_custom-v0',
    entry_point='reacher.envs:ReacherCustomEnv',
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id='reacher_custom-action1-v0',
    entry_point='reacher.envs:ReacherCustomAction1Env',
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id='reacher_custom-action2-v0',
    entry_point='reacher.envs:ReacherCustomAction2Env',
    max_episode_steps=50,
    reward_threshold=-3.75,
)
register(
    id='reacher_custom-raction1-v0',
    entry_point='reacher.envs:ReacherCustomRAction1Env',
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id='reacher_custom-raction2-v0',
    entry_point='reacher.envs:ReacherCustomRAction2Env',
    max_episode_steps=50,
    reward_threshold=-3.75,
)
