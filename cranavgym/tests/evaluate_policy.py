import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor,
    is_vecenv_wrapped,
)


# extra for mw
from stable_baselines3.common.utils import obs_as_tensor
import torch as th
import os
from PIL import Image


def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
    log_dir=None,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = (
        is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    )

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_infos = []
    current_episode_infos = []
    episode_values = []
    current_episode_values = []
    episode_features = []
    current_episode_features = []
    episode_states = []
    current_episode_states = []
    episode_actions = []
    current_episode_actions = []

    total_counter = 0

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array(
        [(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int"
    )

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )

        # obs_tensor = obs_as_tensor(self._last_obs, self.device)
        # actions, values, log_probs = self.policy(obs_tensor)

        # NB: this only works for PPO
        # with th.no_grad():
        #     # Convert to pytorch tensor or to TensorDict
        #     # obs_tensor = obs_as_tensor(observations, "cuda")
        #     # actions, values, log_probs = model.policy(obs_tensor)
        #     # convert WHC to CWH
        #     # print(f"{observations.shape=}")

        #     # Below should be enabled for frame stack is false
        #     obs_cwh = np.transpose(observations, (0, 3, 2, 1))
        #     # Below should be enabled for frame stack is true
        #     # obs_cwh = observations
        #     # print(f"{obs_cwh.shape=}")
        #     # get the value, then cast to cpu normal value (from a tensor)
        #     # use below if frame stack is false
        #     features = model.policy.extract_features(obs_as_tensor(obs_cwh, "cuda"))[0]
        #     # use below if frame stack is true
        #     # features = model.policy.extract_features(obs_as_tensor(obs_cwh, "cuda"))
        #     print(f"{features=}")
        #     print(f"{features.shape=}")
        #     value = model.policy.predict_values(obs_as_tensor(obs_cwh, "cuda")).item()
        #     # value = 0
        #     # print(f"{value=}")

        # IF TD3:
        value = 0
        features = None
        lf = None
        current_episode_values.append(value)
        current_episode_features.append(features)
        current_episode_states.append(observations)
        current_episode_actions.append(actions)

        lf_name = f"latent_features{total_counter}.npy"
        state_name = f"state{total_counter}.png"
        # IF PPO:
        # lf = features.detach().cpu().numpy()
        state = observations[0]

        latent_features_name = np.save(os.path.join(log_dir, f"lf/{lf_name}"), lf)
        # # convert state to PIL image and flip BGR>RGB
        im = Image.fromarray(state[:, :, ::-1])
        state_name = os.path.join(log_dir, f"states/{state_name}")
        im.save(state_name)
        total_counter += 1

        # params = model.get_parameters()
        # for key, value in params.items():
        #     print(f"{key=}")
        # for key2, val2 in params['policy'].items():
        #         print(f"{key2=}")

        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        current_episode_infos.append(infos)
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            episode_infos.append(current_episode_infos)
                            episode_values.append(current_episode_values)
                            episode_features.append(current_episode_features)
                            episode_states.append(current_episode_states)
                            episode_actions.append(current_episode_actions)
                            current_episode_infos = []
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, (
            "Mean reward below threshold: "
            f"{mean_reward:.2f} < {reward_threshold:.2f}"
        )
    if return_episode_rewards:
        return (
            episode_rewards,
            episode_lengths,
            episode_infos,
            episode_values,
            # episode_features,
            # episode_states,
            episode_actions,
        )
    return mean_reward, std_reward
