import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.python.summary.summary_iterator import summary_iterator

import matplotlib
import seaborn as sns

matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Bitstream Vera Sans"
matplotlib.rcParams["mathtext.it"] = "Bitstream Vera Sans:italic"
matplotlib.rcParams["mathtext.bf"] = "Bitstream Vera Sans:bold"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


# BAR_COLORS = ["#1B4966", "#83C2EA", "#3CA5E6", "#6CD5A1"]
# colour blind scheme


BAR_COLORS = [
    "#648FFF",
    "#785EF0",
    "#DC267F",
    "#FE6100",
    "#FFB000",
    "#1B4966",
    "#83C2EA",
]
# BAR_COLORS = ["#785EF0", "#FE6100", "#FFB000", "#DC267F"]


# def plot_single_line(fig, steps, reward):
#     # figsize = (6, 4)
#     # -----------------------------------------REWARD PLOT------------------------------------
#     # hypothetical_optimal_human = np.full(len(steps), length_max*HYPOTHETICAL_OPTIMAL_REWARD)

#     plt.figure(figsize=figsize, dpi=600, layout="constrained")
#     plt.plot(
#         steps, reward, color="lightskyblue", label="PPO LSTM Raw Values", alpha=0.5
#     )

#     rewards_smoothed = pd.DataFrame(reward)
#     rewards_smoothed = rewards_smoothed.rolling(window=50).mean()

#     plt.plot(
#         steps, rewards_smoothed, color="royalblue", label="PPO LSTM Running Average"
#     )
#     return


def load_single_line(path):
    steps = []
    rewards = []
    lengths = []
    for e in summary_iterator(path):
        for v in e.summary.value:
            # if v.tag == 'rollout/ep_rew_mean' or v.tag == 'rollout/ep_len_mean':
            if v.tag == "rollout/ep_rew_mean":
                # print(f"{v=}")
                steps.append(e.step)
                rewards.append(v.simple_value)
            elif v.tag == "rollout/ep_len_mean":
                lengths.append(v.simple_value)
    return steps, rewards, lengths


def load_single_line_2mill(path):
    steps = []
    rewards = []
    lengths = []
    for e in summary_iterator(path):
        for v in e.summary.value:
            if e.step < 2000000:
                # if v.tag == 'rollout/ep_rew_mean' or v.tag == 'rollout/ep_len_mean':
                if v.tag == "rollout/ep_rew_mean":
                    # print(f"{v=}")
                    steps.append(e.step)
                    rewards.append(v.simple_value)
                elif v.tag == "rollout/ep_len_mean":
                    lengths.append(v.simple_value)
    return steps, rewards, lengths


def load_single_line_walltime(path):
    steps = []
    rewards = []
    lengths = []
    is_first = True
    first_walltime = 0
    for e in summary_iterator(path):
        for v in e.summary.value:
            # if v.tag == 'rollout/ep_rew_mean' or v.tag == 'rollout/ep_len_mean':
            if v.tag == "rollout/ep_rew_mean":
                if is_first:
                    first_walltime = e.wall_time
                    is_first = False
                # print(f"{e=}")
                steps.append(e.wall_time - first_walltime)
                rewards.append(v.simple_value)
            elif v.tag == "rollout/ep_len_mean":
                lengths.append(v.simple_value)
    return steps, rewards, lengths


def load_dreamer_line(path):
    import tensorflow as tf

    steps = []
    rewards = []
    lengths = []
    for e in summary_iterator(path):
        for v in e.summary.value:
            # if v.tag == 'rollout/ep_rew_mean' or v.tag == 'rollout/ep_len_mean':
            if e.step < 500000:
                if v.tag == "episode/score":
                    # print(f"{e=}")
                    steps.append(e.step)
                    # rewards.append(v.simple_value)
                    rewards.append(tf.make_ndarray(v.tensor))
                elif v.tag == "episode/length":
                    lengths.append(tf.make_ndarray(v.tensor))
    return steps, rewards, lengths


def load_dreamer_line_walltime(path):
    import tensorflow as tf

    steps = []
    rewards = []
    lengths = []
    is_first = True
    first_walltime = 0
    for e in summary_iterator(path):
        for v in e.summary.value:
            # if v.tag == 'rollout/ep_rew_mean' or v.tag == 'rollout/ep_len_mean':
            if e.step < 500000:
                if v.tag == "episode/score":
                    # print(f"{v=}")
                    if is_first:
                        first_walltime = e.wall_time
                        is_first = False
                    # print(f"{e=}")
                    steps.append(e.wall_time - first_walltime)
                    # rewards.append(v.simple_value)
                    rewards.append(tf.make_ndarray(v.tensor))
                elif v.tag == "episode/length":
                    lengths.append(tf.make_ndarray(v.tensor))
    return steps, rewards, lengths


def plot_for_scenario(path, plot_names, scenario_save, scenario_title, length_max=300):
    steps = []
    rewards = []
    # plot_names = []
    colours = []
    steps1, rewards1, _ = load_single_line(path[0])
    steps2, rewards2, _ = load_single_line(path[1])
    steps3, rewards3, _ = load_single_line(path[2])

    steps.append(steps1)
    steps.append(steps2)
    steps.append(steps3)

    rewards.append(rewards1)
    rewards.append(rewards2)
    rewards.append(rewards3)

    if len(path) == 4:
        steps4, rewards4, _ = load_single_line(path[3])
        steps.append(steps4)
        rewards.append(rewards4)
        colours.append("mediumpurple")
        colours.append("lightskyblue")
        colours.append("darkorange")
        colours.append("springgreen")
    else:
        colours.append("lightskyblue")
        colours.append("darkorange")
        colours.append("springgreen")
        colours.append("mediumpurple")

    # for IEEE double column use figsize (6,4)
    # figsize = (6, 4)
    # For springer nature single column use (9,4)
    figsize = (9, 4)
    # -----------------------------------------REWARD PLOT------------------------------------
    max_theoretical_reward = np.full(len(steps1), (1.0))
    max_reward = np.full(len(steps1), (1.0 - (10 / 50)))

    plt.figure(figsize=figsize, dpi=600, layout="constrained")
    for plot_zip in zip(steps, rewards, plot_names, colours):
        plt.plot(
            plot_zip[0], plot_zip[1], color=plot_zip[3], label=plot_zip[2], alpha=1.0
        )

        rewards_smoothed = pd.DataFrame(plot_zip[1])
        rewards_smoothed = rewards_smoothed.rolling(window=50).mean()

        # plt.plot(
        #     plot_zip[0],
        #     rewards_smoothed,
        #     color="royalblue",
        #     label=(plot_zip[2] + " Average"),
        # )

    # plt.plot(steps1, max_reward, color='black', label='Maximum possible reward (assuming 0 steps)', alpha=1.0)
    plt.plot(
        steps1,
        max_reward,
        color="black",
        label="Maximum episode reward (assuming 10 steps to reach the goal)",
        alpha=0.7,
    )

    # Use scientific magnitudes
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    # plt.gca().ticklabel_format(useMathText=True)

    plt.title(f"Mean Episode Reward for the {scenario_title} Scenario")
    plt.xlabel("Step")
    plt.ylabel("Episode Reward")
    plt.legend()
    # plt.show()
    plt.savefig(f"./figures_journal/{scenario_save}_episode_reward.pdf")

    # -----------------------------------------LENGTH PLOT------------------------------------
    # length_max_line = np.full(len(steps), length_max)

    # plt.figure(figsize=figsize, dpi=600, layout="constrained")
    # plt.plot(
    #     steps, lengths, color="lightskyblue", label="PPO LSTM Raw Values", alpha=0.5
    # )

    # lengths_smoothed = pd.DataFrame(lengths)
    # lengths_smoothed = lengths_smoothed.rolling(window=50).mean()

    # plt.plot(
    #     steps, lengths_smoothed, color="royalblue", label="PPO LSTM Running Average"
    # )

    # plt.plot(
    #     steps, length_max_line, color="black", label="Maximum Episode Length", alpha=0.7
    # )

    # index_max_length = max(range(len(lengths)), key=lengths.__getitem__)
    # index_max_reward = max(range(len(rewards)), key=rewards.__getitem__)
    # print(
    #     f"{scenario_title}. {max(lengths)=} occured at {steps[index_max_length]}, with mean reward {rewards[index_max_length]}"
    # )
    # print(
    #     f"{scenario_title}. {max(rewards)=} occured at {steps[index_max_reward]}, with mean length {lengths[index_max_reward]}"
    # )

    # # Use scientific magnitudes
    # plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    # # plt.gca().ticklabel_format(useMathText=True)

    # plt.title(f"Mean Episode Length for the {scenario_title} Scenario")
    # plt.xlabel("Step")
    # plt.ylabel("Episode Length")
    # plt.legend()
    # # plt.show()
    # plt.savefig(f"./figures_journal/{scenario_save}_episode_length.pdf")


# specific for the algo scenario
def plot_algo_scenario(path, plot_names, scenario_save, scenario_title, length_max=300):
    steps = []
    rewards = []
    # plot_names = []
    colours = []
    steps1, rewards1, _ = load_single_line(path[0])
    steps2, rewards2, _ = load_single_line(path[1])
    steps3, rewards3, _ = load_single_line(path[2])
    steps3_5, rewards3_5, _ = load_dreamer_line(path[3])
    steps4, rewards4, _ = load_single_line(path[4])
    steps5, rewards5, _ = load_dreamer_line(path[5])

    rewards5 = pd.DataFrame(rewards5)
    rewards5 = rewards5.rolling(window=32).mean()
    rewards5 = rewards5 * 0.75
    rewards3_5 = pd.DataFrame(rewards3_5)
    rewards3_5 = rewards3_5.rolling(window=32).mean()

    steps.append(steps1)
    steps.append(steps2)
    steps.append(steps3)
    steps.append(steps3_5)
    steps.append(steps4)
    steps.append(steps5)

    rewards.append(rewards1)
    rewards.append(rewards2)
    rewards.append(rewards3)
    rewards.append(rewards3_5)
    rewards.append(rewards4)
    rewards.append(rewards5)

    if len(path) == 4:
        steps4, rewards4, _ = load_single_line(path[3])
        steps.append(steps4)
        rewards.append(rewards4)
        colours.append("mediumpurple")
        colours.append("lightskyblue")
        colours.append("darkorange")
        colours.append("springgreen")
    else:
        # colours.append("lightskyblue")
        # colours.append("darkorange")
        # colours.append("springgreen")
        # colours.append("mediumpurple")
        # BAR_COLORS = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]
        colours.append("mediumpurple")
        colours.append("lightskyblue")
        colours.append("darkorange")
        colours.append("springgreen")
        colours.append("#FFB000")
        colours.append("hotpink")

    # for IEEE double column use figsize (6,4)
    # figsize = (6, 4)
    # For springer nature single column use (9,4)
    figsize = (9, 4)
    # -----------------------------------------REWARD PLOT------------------------------------
    max_theoretical_reward = np.full(len(steps1), (1.0))
    max_reward = np.full(len(steps1), (1.0 - (10 / 50)))

    plt.figure(figsize=figsize, dpi=600, layout="constrained")
    for plot_zip in zip(steps, rewards, plot_names, colours):
        plt.plot(
            plot_zip[0], plot_zip[1], color=plot_zip[3], label=plot_zip[2], alpha=1.0
        )

        rewards_smoothed = pd.DataFrame(plot_zip[1])
        rewards_smoothed = rewards_smoothed.rolling(window=50).mean()

        # plt.plot(
        #     plot_zip[0],
        #     rewards_smoothed,
        #     color="royalblue",
        #     label=(plot_zip[2] + " Average"),
        # )

    # plt.plot(steps1, max_reward, color='black', label='Maximum possible reward (assuming 0 steps)', alpha=1.0)
    plt.plot(
        steps1,
        max_reward,
        color="black",
        label="Maximum episode reward (assuming 10 steps to reach the goal)",
        alpha=0.7,
    )

    # Use scientific magnitudes
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    # plt.gca().ticklabel_format(useMathText=True)

    plt.title(f"Mean Episode Reward for the {scenario_title} Scenario")
    plt.xlabel("Step")
    plt.ylabel("Episode Reward")
    plt.legend()
    # plt.show()
    plt.savefig(f"./figures_journal/{scenario_save}_episode_reward.pdf")


def plot_algo_scenario2(
    path, plot_names, scenario_save, scenario_title, length_max=300
):
    steps = []
    rewards = []
    # plot_names = []
    colours = []
    steps1, rewards1, _ = load_single_line(path[0])
    steps2, rewards2, _ = load_single_line(path[1])
    steps3, rewards3, _ = load_single_line(path[2])
    steps3_5, rewards3_5, _ = load_dreamer_line(path[3])
    # steps4, rewards4, _ = load_single_line(path[4])
    # steps5, rewards5, _ = load_dreamer_line(path[5])

    # rewards5 = pd.DataFrame(rewards5)
    # rewards5 = rewards5.rolling(window=32).mean()
    # rewards5 = rewards5 * 0.75
    rewards3_5 = pd.DataFrame(rewards3_5)
    rewards3_5 = rewards3_5.rolling(window=32).mean()

    steps.append(steps1)
    steps.append(steps2)
    steps.append(steps3)
    steps.append(steps3_5)
    # steps.append(steps4)
    # steps.append(steps5)

    rewards.append(rewards1)
    rewards.append(rewards2)
    rewards.append(rewards3)
    rewards.append(rewards3_5)
    # rewards.append(rewards4)
    # rewards.append(rewards5)

    if len(path) == 4:
        steps4, rewards4, _ = load_single_line(path[3])
        steps.append(steps4)
        rewards.append(rewards4)
        colours.append("mediumpurple")
        colours.append("lightskyblue")
        colours.append("darkorange")
        colours.append("springgreen")
    else:
        # colours.append("lightskyblue")
        # colours.append("darkorange")
        # colours.append("springgreen")
        # colours.append("mediumpurple")
        # BAR_COLORS = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]
        colours.append("mediumpurple")
        colours.append("lightskyblue")
        colours.append("darkorange")
        colours.append("springgreen")
        colours.append("#FFB000")
        colours.append("hotpink")

    # for IEEE double column use figsize (6,4)
    # figsize = (6, 4)
    # For springer nature single column use (9,4)
    figsize = (9, 4)
    # -----------------------------------------REWARD PLOT------------------------------------
    max_theoretical_reward = np.full(len(steps1), (1.0))
    max_reward = np.full(len(steps1), (1.0 - (10 / 50)))

    plt.figure(figsize=figsize, dpi=600, layout="constrained")
    for plot_zip in zip(steps, rewards, plot_names, colours):
        plt.plot(
            plot_zip[0], plot_zip[1], color=plot_zip[3], label=plot_zip[2], alpha=1.0
        )

        rewards_smoothed = pd.DataFrame(plot_zip[1])
        rewards_smoothed = rewards_smoothed.rolling(window=50).mean()

        # plt.plot(
        #     plot_zip[0],
        #     rewards_smoothed,
        #     color="royalblue",
        #     label=(plot_zip[2] + " Average"),
        # )

    # plt.plot(steps1, max_reward, color='black', label='Maximum possible reward (assuming 0 steps)', alpha=1.0)
    plt.plot(
        steps1,
        max_reward,
        color="black",
        label="Maximum episode reward (assuming 10 steps to reach the goal)",
        alpha=0.7,
    )

    # Use scientific magnitudes
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    # plt.gca().ticklabel_format(useMathText=True)

    plt.title(f"Mean Episode Reward for the {scenario_title} Scenario")
    plt.xlabel("Step")
    plt.ylabel("Episode Reward")
    plt.legend()
    # plt.show()
    plt.savefig(f"./figures_journal/{scenario_save}_episode_reward.pdf")


def plot_algo_scenario3(
    path, plot_names, scenario_save, scenario_title, length_max=300
):
    steps = []
    rewards = []
    # plot_names = []
    colours = []
    steps1, rewards1, _ = load_single_line(path[0])
    steps2, rewards2, _ = load_single_line(path[1])
    steps3, rewards3, _ = load_single_line(path[2])
    steps4, rewards4, _ = load_dreamer_line(path[3])
    # steps4, rewards4, _ = load_single_line(path[4])
    # steps5, rewards5, _ = load_dreamer_line(path[5])

    # rewards5 = pd.DataFrame(rewards5)
    # rewards5 = rewards5.rolling(window=32).mean()
    rewards4 = pd.DataFrame(rewards4)
    # rewards2 = (
    #     rewards2 * 0.75
    # )  # nb: Dreamer rewards need to be scaled (old reward) -> nb needs rerun on updated env with new reward, keep as a placeholder
    rewards4 = rewards4.rolling(window=128).mean()

    steps.append(steps1)
    steps.append(steps2)
    steps.append(steps3)
    # steps.append(steps3_5)
    steps.append(steps4)
    # steps.append(steps5)

    rewards.append(rewards1)
    rewards.append(rewards2)
    rewards.append(rewards3)
    # rewards.append(rewards3_5)
    rewards.append(rewards4)
    # rewards.append(rewards5)

    if len(path) == 4:
        # steps4, rewards4, _ = load_single_line(path[3])
        # steps.append(steps4)
        # rewards.append(rewards4)
        colours.append("mediumpurple")
        colours.append("lightskyblue")
        colours.append("darkorange")
        colours.append("springgreen")
    else:
        # colours.append("lightskyblue")
        # colours.append("darkorange")
        # colours.append("springgreen")
        # colours.append("mediumpurple")
        # BAR_COLORS = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]
        colours.append("mediumpurple")
        colours.append("lightskyblue")
        colours.append("darkorange")
        colours.append("springgreen")
        colours.append("#FFB000")
        colours.append("hotpink")

    # for IEEE double column use figsize (6,4)
    # figsize = (6, 4)
    # For springer nature single column use (9,4)
    figsize = (9, 4)
    # -----------------------------------------REWARD PLOT------------------------------------
    max_theoretical_reward = np.full(len(steps1), (1.0))
    max_reward = np.full(len(steps1), (1.0 - (100 / 500)))

    plt.figure(figsize=figsize, dpi=600, layout="constrained")
    for plot_zip in zip(steps, rewards, plot_names, colours):
        plt.plot(
            plot_zip[0], plot_zip[1], color=plot_zip[3], label=plot_zip[2], alpha=1.0
        )

        rewards_smoothed = pd.DataFrame(plot_zip[1])
        rewards_smoothed = rewards_smoothed.rolling(window=50).mean()

        # plt.plot(
        #     plot_zip[0],
        #     rewards_smoothed,
        #     color="royalblue",
        #     label=(plot_zip[2] + " Average"),
        # )

    # plt.plot(steps1, max_reward, color='black', label='Maximum possible reward (assuming 0 steps)', alpha=1.0)
    plt.plot(
        steps1,
        max_reward,
        color="black",
        label="Maximum episode reward (assuming 100 steps to reach the goal)",
        alpha=0.7,
    )

    # Use scientific magnitudes
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    # plt.gca().ticklabel_format(useMathText=True)

    plt.title(f"Mean Episode Reward for the {scenario_title} Scenario")
    plt.xlabel("Step")
    plt.ylabel("Episode Reward")
    plt.legend()
    # plt.show()
    plt.savefig(f"./figures_journal/{scenario_save}_episode_reward.pdf")


def plot_algo_scenario4_wall_time(
    path, plot_names, scenario_save, scenario_title, length_max=300
):
    steps = []
    rewards = []
    # plot_names = []
    colours = []
    steps1, rewards1, _ = load_single_line_walltime(path[0])
    steps2, rewards2, _ = load_single_line_walltime(path[1])
    steps3, rewards3, _ = load_single_line_walltime(path[2])
    steps4, rewards4, _ = load_dreamer_line_walltime(path[3])
    # steps4, rewards4, _ = load_single_line(path[4])
    # steps5, rewards5, _ = load_dreamer_line(path[5])

    # rewards5 = pd.DataFrame(rewards5)
    # rewards5 = rewards5.rolling(window=32).mean()
    rewards4 = pd.DataFrame(rewards4)
    # rewards2 = (
    #     rewards2 * 0.75
    # )  # nb: Dreamer rewards need to be scaled (old reward) -> nb needs rerun on updated env with new reward, keep as a placeholder
    rewards4 = rewards4.rolling(window=128).mean()

    steps.append(steps1)
    steps.append(steps2)
    steps.append(steps3)
    # steps.append(steps3_5)
    steps.append(steps4)
    # steps.append(steps5)

    rewards.append(rewards1)
    rewards.append(rewards2)
    rewards.append(rewards3)
    # rewards.append(rewards3_5)
    rewards.append(rewards4)
    # rewards.append(rewards5)

    if len(path) == 4:
        # steps4, rewards4, _ = load_single_line(path[3])
        # steps.append(steps4)
        # rewards.append(rewards4)
        colours.append("mediumpurple")
        colours.append("lightskyblue")
        colours.append("darkorange")
        colours.append("springgreen")
    else:
        # colours.append("lightskyblue")
        # colours.append("darkorange")
        # colours.append("springgreen")
        # colours.append("mediumpurple")
        # BAR_COLORS = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]
        colours.append("mediumpurple")
        colours.append("lightskyblue")
        colours.append("darkorange")
        colours.append("springgreen")
        colours.append("#FFB000")
        colours.append("hotpink")

    # for IEEE double column use figsize (6,4)
    # figsize = (6, 4)
    # For springer nature single column use (9,4)
    figsize = (9, 4)
    # -----------------------------------------REWARD PLOT------------------------------------
    max_theoretical_reward = np.full(len(steps1), (1.0))
    max_reward = np.full(len(steps1), (1.0 - (100 / 500)))

    plt.figure(figsize=figsize, dpi=600, layout="constrained")
    for plot_zip in zip(steps, rewards, plot_names, colours):
        plt.plot(
            plot_zip[0], plot_zip[1], color=plot_zip[3], label=plot_zip[2], alpha=1.0
        )

        rewards_smoothed = pd.DataFrame(plot_zip[1])
        rewards_smoothed = rewards_smoothed.rolling(window=50).mean()

        # plt.plot(
        #     plot_zip[0],
        #     rewards_smoothed,
        #     color="royalblue",
        #     label=(plot_zip[2] + " Average"),
        # )

    # plt.plot(steps1, max_reward, color='black', label='Maximum possible reward (assuming 0 steps)', alpha=1.0)
    plt.plot(
        steps1,
        max_reward,
        color="black",
        label="Maximum episode reward (assuming 100 steps to reach the goal)",
        alpha=0.7,
    )

    # Use scientific magnitudes
    # plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    # plt.gca().ticklabel_format(useMathText=True)

    plt.title(f"Mean Episode Reward for the {scenario_title} Scenario")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Episode Reward")
    plt.legend()
    # plt.show()
    plt.savefig(f"./figures_journal/{scenario_save}_episode_reward.pdf")


def plot_algo_scenario5(
    path, plot_names, scenario_save, scenario_title, length_max=300
):
    steps = []
    rewards = []
    # plot_names = []
    colours = []
    steps1, rewards1, _ = load_single_line(path[0])
    steps2, rewards2, _ = load_single_line(path[1])
    steps3, rewards3, _ = load_single_line(path[2])
    steps4, rewards4, _ = load_single_line_2mill(path[3])
    steps5, rewards5, _ = load_single_line(path[4])
    steps6, rewards6, _ = load_single_line(path[5])
    steps7, rewards7, _ = load_dreamer_line(path[6])
    steps8, rewards8, _ = load_dreamer_line(path[7])
    # steps4, rewards4, _ = load_single_line(path[4])
    # steps5, rewards5, _ = load_dreamer_line(path[5])

    # rewards5 = pd.DataFrame(rewards5)
    # rewards5 = rewards5.rolling(window=32).mean()
    rewards7 = pd.DataFrame(rewards7)
    rewards8 = pd.DataFrame(rewards8)
    # rewards2 = (
    #     rewards2 * 0.75
    # )  # nb: Dreamer rewards need to be scaled (old reward) -> nb needs rerun on updated env with new reward, keep as a placeholder
    rewards7 = rewards7.rolling(window=128).mean()
    rewards8 = rewards8.rolling(window=128).mean()

    steps.append(steps1)
    steps.append(steps2)
    steps.append(steps3)
    # steps.append(steps3_5)
    steps.append(steps4)
    steps.append(steps5)
    steps.append(steps6)
    steps.append(steps7)
    steps.append(steps8)

    rewards.append(rewards1)
    rewards.append(rewards2)
    rewards.append(rewards3)
    # rewards.append(rewards3_5)
    rewards.append(rewards4)
    rewards.append(rewards5)
    rewards.append(rewards6)
    rewards.append(rewards7)
    rewards.append(rewards8)
    # rewards.append(rewards5)

    if len(path) == 4:
        # steps4, rewards4, _ = load_single_line(path[3])
        # steps.append(steps4)
        # rewards.append(rewards4)
        colours.append("mediumpurple")
        colours.append("lightskyblue")
        colours.append("darkorange")
        colours.append("springgreen")
    else:
        # colours.append("lightskyblue")
        # colours.append("darkorange")
        # colours.append("springgreen")
        # colours.append("mediumpurple")
        # BAR_COLORS = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]
        colours.append("mediumpurple")
        colours.append("lightskyblue")
        colours.append("darkorange")
        colours.append("springgreen")
        colours.append("#FFB000")
        colours.append("hotpink")
        colours.append("#648FFF")
        colours.append("#785EF0")

    # for IEEE double column use figsize (6,4)
    # figsize = (6, 4)
    # For springer nature single column use (9,4)
    figsize = (9, 4)
    # -----------------------------------------REWARD PLOT------------------------------------
    max_theoretical_reward = np.full(len(steps1), (1.0))
    max_reward = np.full(len(steps4), (1.0 - (100 / 500)))

    plt.figure(figsize=figsize, dpi=600, layout="constrained")
    for plot_zip in zip(steps, rewards, plot_names, colours):
        plt.plot(
            plot_zip[0], plot_zip[1], color=plot_zip[3], label=plot_zip[2], alpha=1.0
        )

        rewards_smoothed = pd.DataFrame(plot_zip[1])
        rewards_smoothed = rewards_smoothed.rolling(window=50).mean()

        # plt.plot(
        #     plot_zip[0],
        #     rewards_smoothed,
        #     color="royalblue",
        #     label=(plot_zip[2] + " Average"),
        # )

    # plt.plot(steps1, max_reward, color='black', label='Maximum possible reward (assuming 0 steps)', alpha=1.0)
    plt.plot(
        steps4,
        max_reward,
        color="black",
        label="Maximum episode reward (assuming 100 steps to reach the goal)",
        alpha=0.7,
    )

    # Use scientific magnitudes
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    # plt.gca().ticklabel_format(useMathText=True)

    plt.title(f"Mean Episode Reward for the {scenario_title}")
    plt.xlabel("Step")
    plt.ylabel("Episode Reward")
    plt.legend()
    # plt.show()
    plt.savefig(f"./figures_journal/{scenario_save}_episode_reward.pdf")


# def plot_noise_graph_lidar(save_name, title):
#     x_labels = ("0x0", "3x3", "5x5", "7x7")

#     td3_results = (0.32, -0.04, -0.81, -0.52)
#     ppo_results = (0.22, -0.48, -0.43, -0.55)
#     # ppo_lstm_results = (-0.6, -1.0, -1.0, -1.0)
#     dreamerv3_results = (0.32, -0.56, -0.69, -0.25)

#     evaluation_vals = {
#         "TD3": tuple(td3_results),
#         "PPO": tuple(ppo_results),
#         # "PPO-LSTM": tuple(ppo_lstm_results),
#         "DreamerV3": tuple(dreamerv3_results),
#     }

#     plot_bar_chart(x_labels, evaluation_vals, save_name, title)


def plot_noise_graph_lidar_0x0(save_name, title):
    x_labels = ("0x0", "3x3", "5x5", "7x7")

    td3_results = (87.2, 77.0, 67.0, 51.4)
    ppo_results = (83.0, 70.0, 46.6, 35.2)
    # ppo_lstm_results = (-0.6, -1.0, -1.0, -1.0)
    # dreamerv3_results = (0.32, -0.56, -0.69, -0.25)

    td3_std = (3.31, 4.94, 5.02, 4.76)
    ppo_std = (2.53, 4.24, 4.32, 4.83)
    # dreamer_std =(0,0,0,0)

    evaluation_vals = {
        "TD3": tuple(td3_results),
        "PPO": tuple(ppo_results),
        # "PPO-LSTM": tuple(ppo_lstm_results),
        # "DreamerV3": tuple(dreamerv3_results),
    }
    errors = [td3_std, ppo_std]

    plot_bar_chart(x_labels, evaluation_vals, save_name, title, errors=errors)


def plot_noise_graph_lidar_3x3(save_name, title):
    x_labels = ("0x0", "3x3", "5x5", "7x7")

    td3_results = (69.6, 64.0, 54.8, 47.2)
    ppo_results = (68.0, 65.2, 58.2, 54.0)
    # ppo_lstm_results = (-0.6, -1.0, -1.0, -1.0)
    # dreamerv3_results = (0.32, -0.56, -0.69, -0.25)

    td3_std = (3.01, 6.10, 6.34, 2.48)
    ppo_std = (3.03, 4.92, 4.02, 4.29)
    # dreamer_std =(0,0,0,0)

    evaluation_vals = {
        "TD3": tuple(td3_results),
        "PPO": tuple(ppo_results),
        # "PPO-LSTM": tuple(ppo_lstm_results),
        # "DreamerV3": tuple(dreamerv3_results),
    }
    errors = [td3_std, ppo_std]

    plot_bar_chart(x_labels, evaluation_vals, save_name, title, errors=errors)


# Plot for models trained on 3x3 areas of noise
def plot_noise_graph_lidar_ppo_line_error(save_name, title):
    x_labels = ("0x0", "3x3", "5x5", "7x7")

    ppo_results_0x0 = (83.0, 70.0, 46.6, 35.2)
    ppo_results_3x3 = (68.0, 65.2, 58.2, 54.0)
    ppo_results_5x5 = (59.0, 58.6, 55.0, 51.2)
    ppo_results_7x7 = (58.8, 63.4, 60.2, 50.6)
    ppo_results_0x0_std = (2.53, 4.24, 4.32, 4.83)
    ppo_results_3x3_std = (3.03, 4.92, 4.02, 4.29)
    ppo_results_5x5_std = (5.06, 4.63, 4.23, 2.40)
    ppo_results_7x7_std = (6.46, 4.76, 2.14, 3.88)
    # ppo_lstm_results = (76.0, 61.0, 23.0, 3.0)
    # dreamerv3_results = (0.0, 0.0, 0.0, -0.0)

    evaluation_vals = {
        "PPO (0x0)": tuple(ppo_results_0x0),
        "PPO (3x3)": tuple(ppo_results_3x3),
        "PPO (5x5)": tuple(ppo_results_5x5),
        "PPO (7x7)": tuple(ppo_results_7x7),
    }

    errors = [
        ppo_results_0x0_std,
        ppo_results_3x3_std,
        ppo_results_5x5_std,
        ppo_results_7x7_std,
    ]
    plot_line_chart(x_labels, evaluation_vals, save_name, title, errors=errors)


# Plot for models trained on 3x3 areas of noise
def plot_noise_graph_lidar_td3_line_error(save_name, title):
    x_labels = ("0x0", "3x3", "5x5", "7x7")

    td3_results_0x0 = (87.2, 77.0, 67.0, 51.4)
    td3_results_3x3 = (69.6, 64.0, 54.8, 47.2)
    td3_results_5x5 = (8.4, 7.4, 17.6, 25.2)
    td3_results_7x7 = (26.8, 32.0, 35.2, 40.2)
    td3_results_0x0_std = (3.31, 4.94, 5.02, 4.76)
    td3_results_3x3_std = (3.01, 6.10, 6.34, 2.48)
    td3_results_5x5_std = (1.85, 2.15, 1.36, 5.70)
    td3_results_7x7_std = (4.02, 2.53, 3.49, 4.83)
    # ppo_lstm_results = (76.0, 61.0, 23.0, 3.0)
    # dreamerv3_results = (0.0, 0.0, 0.0, -0.0)

    evaluation_vals = {
        "TD3 (0x0)": tuple(td3_results_0x0),
        "TD3 (3x3)": tuple(td3_results_3x3),
        "TD3 (5x5)": tuple(td3_results_5x5),
        "TD3 (7x7)": tuple(td3_results_7x7),
    }

    errors = [
        td3_results_0x0_std,
        td3_results_3x3_std,
        td3_results_5x5_std,
        td3_results_7x7_std,
    ]
    plot_line_chart(x_labels, evaluation_vals, save_name, title, errors=errors)


# ------------------------------CAMERA BELOW--------------------------------


def plot_noise_graph_camera_0x0(save_name, title):
    x_labels = ("0x0", "3x3", "5x5", "7x7")

    # old results
    # td3_results = (0.0, 0.0, 0.0, -0.0)
    # ppo_results = (0.64, -0.10, -0.96, -0.71)
    # # ppo_results_std = (0.64, -0.21, -0.17, -0.18)
    # ppo_lstm_results = (0.0, -0.21, -0.20, -0.19)
    # dreamerv3_results = (0.81, -0.52, -0.68, -0.88)

    # new results: success rate as %
    td3_results = (88.8, 68.4, 17.2, 1.6)
    ppo_results = (71.6, 51.4, 14.0, 1.8)
    # ppo_results_std = (0.64, -0.21, -0.17, -0.18)
    ppo_lstm_results = (82.2, 60.2, 21.0, 1.6)
    dreamerv3_results = (90.8, 73.2, 29.0, 3.0)

    td3_std = (2.93, 4.84, 5.20, 1.2)
    ppo_std = (3.01, 5.08, 3.41, 1.17)
    ppo_lstm_std = (3.19, 2.4, 4.29, 0.8)
    dreamer_std = (2.14, 6.80, 0, 0)

    evaluation_vals = {
        "TD3": tuple(td3_results),
        "PPO": tuple(ppo_results),
        "PPO-LSTM": tuple(ppo_lstm_results),
        "DreamerV3": tuple(dreamerv3_results),
    }
    errors = [td3_std, ppo_std, ppo_lstm_std, dreamer_std]

    plot_bar_chart(x_labels, evaluation_vals, save_name, title, errors=errors)


# Plot for models trained on 3x3 areas of noise
def plot_noise_graph_camera_3x3(save_name, title):
    x_labels = ("0x0", "3x3", "5x5", "7x7")

    # old results
    # td3_results = (0.0, 0.0, 0.0, -0.0)
    # ppo_results = (0.64, -0.10, -0.96, -0.71)
    # # ppo_results_std = (0.64, -0.21, -0.17, -0.18)
    # ppo_lstm_results = (0.0, -0.21, -0.20, -0.19)
    # dreamerv3_results = (0.81, -0.52, -0.68, -0.88)

    # new results: success rate as %
    td3_results = (82.4, 73.2, 45.0, 12.2)
    # ppo_results = (61, 49, 16, 17)#these are from static noise zone run.
    ppo_results = (71.0, 69.2, 41.8, 13.6)
    # ppo_results_std = (0.64, -0.21, -0.17, -0.18)
    ppo_lstm_results = (72.4, 58.6, 36.6, 12.2)
    dreamerv3_results = (86.4, 80.6, 40.0, 15.0)

    td3_std = (4.18, 2.79, 4.24, 3.19)
    ppo_std = (5.83, 2.64, 6.24, 2.58)
    ppo_lstm_std = (3.50, 7.84, 4.27, 1.94)
    dreamer_std = (2.06, 4.13, 0, 0)

    evaluation_vals = {
        "TD3": tuple(td3_results),
        "PPO": tuple(ppo_results),
        "PPO-LSTM": tuple(ppo_lstm_results),
        "DreamerV3": tuple(dreamerv3_results),
    }

    errors = [td3_std, ppo_std, ppo_lstm_std, dreamer_std]
    plot_bar_chart(x_labels, evaluation_vals, save_name, title, errors=errors)


def plot_noise_graph_camera_0x0_randomgoal(save_name, title):
    x_labels = ("0x0", "3x3", "5x5", "7x7")

    # old results

    # new results: success rate as %
    td3_results = (83.0, 41.0, 24.0, 15.0)
    ppo_results = (96.0, 86.0, 66.0, 34.0)
    # ppo_results_std = (0.64, -0.21, -0.17, -0.18)
    ppo_lstm_results = (83.0, 50.0, 24.0, 14.0)
    dreamerv3_results = (99.0, 71.0, 39.0, 19.0)

    td3_std = (0.0, 0.0, 0.0, 0.0)
    ppo_std = (0.0, 0.0, 0.0, 0.0)
    ppo_lstm_std = (0.0, 0.0, 0.0, 0.0)
    dreamer_std = (0.0, 0.0, 0.0, 0.0)

    evaluation_vals = {
        "TD3": tuple(td3_results),
        "PPO": tuple(ppo_results),
        "PPO-LSTM": tuple(ppo_lstm_results),
        "DreamerV3": tuple(dreamerv3_results),
    }
    errors = [td3_std, ppo_std, ppo_lstm_std, dreamer_std]

    plot_bar_chart(x_labels, evaluation_vals, save_name, title, errors=errors)


# Plot for models trained on 3x3 areas of noise
def plot_noise_graph_camera_3x3_randomgoal(save_name, title):
    x_labels = ("0x0", "3x3", "5x5", "7x7")

    # old results
    # td3_results = (0.0, 0.0, 0.0, 0.0)
    ppo_results = (94.0, 81.0, 72.0, 45.0)
    # # ppo_results_std = (0.64, -0.21, -0.17, -0.18)
    # ppo_lstm_results = (0.0, 0.0, 0.0, 0.0)
    dreamerv3_results = (100.0, 97.0, 79.0, 48.0)

    # new results: success rate as %
    # td3_results = (82.4, 73.2, 45.0, 12.2)
    # ppo_results = (61, 49, 16, 17)#these are from static noise zone run.
    # ppo_results = (94.0, 86.0, 77.0, 59.0)
    # ppo_results_std = (0.64, -0.21, -0.17, -0.18)
    # ppo_lstm_results = (72.4, 58.6, 36.6, 12.2)
    # dreamerv3_results = (86.4, 80.6, 40.0, 15.0)

    # td3_std = (0.0, 0.0, 0.0, 0.0)
    ppo_std = (0.0, 0.0, 0.0, 0.0)
    # ppo_lstm_std = (0.0, 0.0, 0.0, 0.0)
    dreamer_std = (0.0, 0.0, 0.0, 0.0)

    evaluation_vals = {
        # "TD3": tuple(td3_results),
        "PPO": tuple(ppo_results),
        # "PPO-LSTM": tuple(ppo_lstm_results),
        "DreamerV3": tuple(dreamerv3_results),
    }

    # errors = [td3_std, ppo_std, ppo_lstm_std, dreamer_std]
    errors = [ppo_std, dreamer_std]
    plot_bar_chart(x_labels, evaluation_vals, save_name, title, errors=errors)


def plot_noise_graph_camera_0x0_no_obstacles(save_name, title):
    x_labels = ("0x0", "3x3", "5x5", "7x7")

    # new results: success rate as %
    td3_results = (99, 3, 2, 0)
    ppo_results = (99, 4, 2, 4)
    # ppo_results_std = (0.64, -0.21, -0.17, -0.18)
    ppo_lstm_results = (98, 3, 2, 2)
    dreamerv3_results = (94, 0, 0, 0)

    evaluation_vals = {
        "TD3": tuple(td3_results),
        "PPO": tuple(ppo_results),
        "PPO-LSTM": tuple(ppo_lstm_results),
        "DreamerV3": tuple(dreamerv3_results),
    }

    plot_bar_chart(x_labels, evaluation_vals, save_name, title)


# Plot for models trained on 3x3 areas of noise
def plot_noise_graph_camera_3x3_no_obstacles(save_name, title):
    x_labels = ("0x0", "3x3", "5x5", "7x7")

    # new results: success rate as %
    td3_results = (98, 74, 14, 13)
    # ppo_results = (61, 49, 16, 17)#these are from static noise zone run.
    ppo_results = (98, 71, 38, 10)
    # ppo_results_std = (0.64, -0.21, -0.17, -0.18)
    ppo_lstm_results = (84, 66, 49, 19)
    dreamerv3_results = (96, 87, 50, 20)  # redo 82??

    evaluation_vals = {
        "TD3": tuple(td3_results),
        "PPO": tuple(ppo_results),
        "PPO-LSTM": tuple(ppo_lstm_results),
        "DreamerV3": tuple(dreamerv3_results),
    }

    plot_bar_chart(x_labels, evaluation_vals, save_name, title)


# Plot for models trained on 3x3 areas of noise
def plot_noise_graph_camera_ppo_line(save_name, title):
    x_labels = ("0x0", "3x3", "5x5", "7x7")

    # old results
    # td3_results = (0.0, 0.0, 0.0, -0.0)
    # ppo_results = (0.64, -0.10, -0.96, -0.71)
    # # ppo_results_std = (0.64, -0.21, -0.17, -0.18)
    # ppo_lstm_results = (0.0, -0.21, -0.20, -0.19)
    # dreamerv3_results = (0.81, -0.52, -0.68, -0.88)

    # new results: success rate as %
    # td3_results = (92.0, 66.0, 22.0, 5.0)
    # ppo_results = (61, 49, 16, 17)#these are from static noise zone run.
    # ppo_results_0x0 = (74, 50, 12, 0)
    # ppo_results_3x3 = (72, 65, 31, 14)
    # ppo_results_5x5 = (53, 46, 32, 11)
    # ppo_results_7x7 = (21, 11, 12, 11)

    ppo_results_0x0 = (74, 50, 12, 0)
    ppo_results_3x3 = (72, 65, 31, 14)
    ppo_results_5x5 = (53, 46, 32, 11)
    ppo_results_7x7 = (21, 11, 12, 11)
    # ppo_results_std = (0.64, -0.21, -0.17, -0.18)
    # ppo_lstm_results = (76.0, 61.0, 23.0, 3.0)
    # dreamerv3_results = (0.0, 0.0, 0.0, -0.0)

    evaluation_vals = {
        "PPO (0x0)": tuple(ppo_results_0x0),
        "PPO (3x3)": tuple(ppo_results_3x3),
        "PPO (5x5)": tuple(ppo_results_5x5),
        "PPO (7x7)": tuple(ppo_results_7x7),
    }

    plot_line_chart(x_labels, evaluation_vals, save_name, title)


# Plot for models trained on 3x3 areas of noise
def plot_noise_graph_camera_ppo_line_error(save_name, title):
    x_labels = ("0x0", "3x3", "5x5", "7x7")

    ppo_results_0x0 = (71.6, 51.4, 14.0, 1.8)
    ppo_results_3x3 = (71.0, 69.2, 41.8, 13.6)
    ppo_results_0x0_std = (3.0, 5.0, 3.4, 1.2)
    ppo_results_3x3_std = (5.8, 2.6, 6.2, 2.6)
    # ppo_lstm_results = (76.0, 61.0, 23.0, 3.0)
    # dreamerv3_results = (0.0, 0.0, 0.0, -0.0)

    evaluation_vals = {
        "PPO (0x0)": tuple(ppo_results_0x0),
        "PPO (3x3)": tuple(ppo_results_3x3),
        # "PPO (5x5)": tuple(ppo_results_5x5),
        # "PPO (7x7)": tuple(ppo_results_7x7),
    }

    errors = [ppo_results_0x0_std, ppo_results_3x3_std]
    plot_line_chart(x_labels, evaluation_vals, save_name, title, errors=errors)

    # Plot for models trained on 3x3 areas of noise


def plot_noise_graph_camera_ppo_lstm_line(save_name, title):
    x_labels = ("0x0", "3x3", "5x5", "7x7")

    ppo_lstm_results_0x0 = (86, 64, 21, 3)
    ppo_lstm_results_3x3 = (72, 62, 33, 18)
    ppo_lstm_results_5x5 = (40, 31, 24, 15)
    ppo_lstm_results_7x7 = (16, 14, 9, 8)
    # ppo_results_std = (0.64, -0.21, -0.17, -0.18)
    # ppo_lstm_results = (76.0, 61.0, 23.0, 3.0)
    # dreamerv3_results = (0.0, 0.0, 0.0, -0.0)

    evaluation_vals = {
        "PPO LSTM (0x0)": tuple(ppo_lstm_results_0x0),
        "PPO LSTM (3x3)": tuple(ppo_lstm_results_3x3),
        "PPO LSTM (5x5)": tuple(ppo_lstm_results_5x5),
        "PPO LSTM (7x7)": tuple(ppo_lstm_results_7x7),
    }

    plot_line_chart(x_labels, evaluation_vals, save_name, title)

    # Plot for models trained on 3x3 areas of noise


def plot_noise_graph_camera_td3_line(save_name, title):
    x_labels = ("0x0", "3x3", "5x5", "7x7")

    td3_results_0x0 = (88, 72, 23, 2)
    td3_results_3x3 = (83, 76, 47, 12)
    td3_results_5x5 = (77, 66, 32, 13)
    td3_results_7x7 = (2, 3, 5, 7)
    # ppo_results_std = (0.64, -0.21, -0.17, -0.18)
    # ppo_lstm_results = (76.0, 61.0, 23.0, 3.0)
    # dreamerv3_results = (0.0, 0.0, 0.0, -0.0)

    evaluation_vals = {
        "TD3 (0x0)": tuple(td3_results_0x0),
        "TD3 (3x3)": tuple(td3_results_3x3),
        "TD3 (5x5)": tuple(td3_results_5x5),
        "TD3 (7x7)": tuple(td3_results_7x7),
    }

    plot_line_chart(x_labels, evaluation_vals, save_name, title)


def plot_noise_graph_camera_ppo_line_no_obstacle_map(save_name, title):
    x_labels = ("0x0", "3x3", "5x5", "7x7")

    ppo_results_0x0 = (99, 4, 2, 4)
    ppo_results_3x3 = (98, 71, 38, 10)
    ppo_results_5x5 = (44, 16, 12, 12)
    ppo_results_7x7 = (14, 15, 11, 12)

    evaluation_vals = {
        "PPO (0x0)": tuple(ppo_results_0x0),
        "PPO (3x3)": tuple(ppo_results_3x3),
        "PPO (5x5)": tuple(ppo_results_5x5),
        "PPO (7x7)": tuple(ppo_results_7x7),
    }

    plot_line_chart(x_labels, evaluation_vals, save_name, title)


def plot_noise_graph_camera_ppo_lstm_no_obstacle_map(save_name, title):
    x_labels = ("0x0", "3x3", "5x5", "7x7")

    ppo_lstm_results_0x0 = (98, 3, 2, 2)
    ppo_lstm_results_3x3 = (84, 66, 49, 19)
    ppo_lstm_results_5x5 = (19, 29, 19, 13)
    ppo_lstm_results_7x7 = (15, 14, 10, 12)

    evaluation_vals = {
        "PPO LSTM (0x0)": tuple(ppo_lstm_results_0x0),
        "PPO LSTM (3x3)": tuple(ppo_lstm_results_3x3),
        "PPO LSTM (5x5)": tuple(ppo_lstm_results_5x5),
        "PPO LSTM (7x7)": tuple(ppo_lstm_results_7x7),
    }

    plot_line_chart(x_labels, evaluation_vals, save_name, title)


def plot_noise_graph_camera_td3_no_obstacle_map(save_name, title):
    x_labels = ("0x0", "3x3", "5x5", "7x7")

    td3_results_0x0 = (99, 3, 2, 0)
    td3_results_3x3 = (98, 74, 14, 13)
    td3_results_5x5 = (92, 79, 44, 22)
    td3_results_7x7 = (4, 7, 3, 4)

    evaluation_vals = {
        "TD3 (0x0)": tuple(td3_results_0x0),
        "TD3 (3x3)": tuple(td3_results_3x3),
        "TD3 (5x5)": tuple(td3_results_5x5),
        "TD3 (7x7)": tuple(td3_results_7x7),
    }

    plot_line_chart(x_labels, evaluation_vals, save_name, title)


def plot_noise_graph_camera_violin(save_name, title):
    # x_labels = ("0x0", "3x3", "5x5", "7x7")
    x_labels = ("0x0", "3x3")
    episode_rewards_ppo = [
        -1.1799999959766865,
        -1.3999999910593033,
        0.8400000035762787,
        0.7200000062584877,
        1.0,
    ]
    episode_rewards_td3 = [
        0.69999959766865,
        -1.3999999910593033,
        0.4400000035762787,
        0.2200000062584877,
        0.7,
    ]
    episode_lengths = [10, 21, 9, 15, 1]

    evaluation_vals = {
        "TD3": tuple(episode_rewards_td3),
        "PPO": tuple(episode_rewards_ppo),
        # "PPO-LSTM": tuple(ppo_lstm_results),
        # "DreamerV3": tuple(dreamerv3_results),
    }
    # evaluation_vals = [sorted(episode_rewards_td3), sorted(episode_rewards_ppo)]

    # convert to dataframe

    d = {
        "reward": [
            0.640000008046627,
            -1.0199999995529652,
            -1.2999999932944775,
            -0.9999999776482582,
            0.30000001564621925,
            -1.1399999968707561,
            0.6000000089406967,
            0.7200000062584877,
            0.7200000062584877,
            -1.3999999910593033,
            -1.1399999968707561,
            0.8200000040233135,
            0.760000005364418,
            0.6000000089406967,
            1.0,
            -0.9999999776482582,
            -0.9999999776482582,
            0.6800000071525574,
            0.4400000125169754,
            -1.1599999964237213,
            0.5000000111758709,
            0.6800000071525574,
            -1.4399999901652336,
            0.6600000075995922,
            0.9200000017881393,
            -1.1399999968707561,
            0.7200000062584877,
            -1.119999997317791,
            0.6200000084936619,
            -1.3799999915063381,
            0.5200000107288361,
            1.0,
            0.6200000084936619,
            0.5200000107288361,
            0.24000001698732376,
            0.5800000093877316,
            -1.119999997317791,
            0.2600000165402889,
            0.5200000107288361,
            -1.3799999915063381,
            0.8400000035762787,
            -1.4399999901652336,
            0.8400000035762787,
            0.9400000013411045,
            0.5200000107288361,
            0.8000000044703484,
            -1.3799999915063381,
            0.8400000035762787,
            0.7000000067055225,
            0.5800000093877316,
        ],
        "noise size": ["3x3"] * 50,
        "algorithm": [
            "TD3",
        ]
        * 50,
        # "3x3": [
        #     0.69999959766865,
        #     -1.3999999910593033,
        #     0.4400000035762787,
        #     0.2200000062584877,
        #     0.7,
        # ],
    }
    dataframe = pd.DataFrame(data=d)

    plot_violin_chart(x_labels, dataframe, save_name, title)


def plot_violin_chart(
    x_labels, dataframe, bar_chart_name, bar_chart_title, errors=None
):
    labels = x_labels
    x = np.arange(len(labels))  # the label locations
    bar_width = 0.2

    # for MDPI single column use figsize (10,5)
    # fig, ax = plt.subplots(layout='constrained', figsize=(10, 5), dpi=600)
    # for IEEE double column use figsize (6,3)
    # fig, ax = plt.subplots(layout='constrained', figsize=(6, 3), dpi=600)
    # for springer nature single column use figsize (10,5)
    fig, ax = plt.subplots(layout="constrained", figsize=(9, 3), dpi=600)

    plt.grid(True)
    plt.grid(visible=False, axis="x")
    plt.grid(visible=True, which="major", axis="y", linestyle="-")
    plt.grid(
        visible=True, which="minor", axis="y", linestyle="-", linewidth="0.5", alpha=0.5
    )

    ax.grid(True)
    # ax.minorticks_on()
    ax.set_axisbelow(True)

    # BAR_COLORS = ["#1B4966", "#83C2EA", "#3CA5E6", "#6CD5A1"]
    i = 0
    # ax.violinplot(evaluation_vals, showmeans=True, showmedians=True, showextrema=True)
    sns.violinplot(data=dataframe, x="noise size", y="reward", cut=0, inner="stick")
    # g = sns.catplot(
    #     data=dataframe, x="noise size", y="reward", kind="violin", inner=None
    # )
    # sns.swarmplot(
    #     data=dataframe, x="noise size", y="reward", color="k", size=3, ax=g.ax
    # )

    # Add labels, title and custom x-axis tick labels
    ax.set_ylabel("$\mathregular{Mean\ Evaluation\ Reward}$")
    ax.set_xlabel("$\mathregular{Noise\ Area\ Size}$")
    ax.set_title(bar_chart_title)
    ax.set_xticks((x + bar_width), labels=labels)
    ax.legend(loc="lower right", ncols=4)
    # ax.set_ylim(0, 1)
    # ax.set_yticks(np.arange(-1.1, 1.1, 0.2))
    # ax.set_ylim(-1.0, 1.0)
    # ax.set_yticks(np.arange(0, 101, 10))
    # plt.show()
    plt.savefig(f"figures_journal/{bar_chart_name}.pdf", dpi=600)
    return


def plot_noise_graph_camera_temp(save_name, title):
    x_labels = ("0x0", "3x3", "5x5", "7x7")

    td3_results = (0.51, 0.37, -0.70, -0.73)
    td3_results_std = (0.0, 0.82, 1.05, 0.99)  # rerun TD3 eval after 100k!
    ppo_results = (0.64, -0.10, -0.96, -0.71)
    ppo_results_std = (0.64, 0.21, 0.17, 0.18)
    ppo_lstm_results = (0.42, -0.32, -0.79, -0.65)
    ppo_lstm_results_std = (0.79, 1.04, 0.69, 0.80)
    dreamerv3_results = (0.81, -0.52, -0.68, -0.88)
    dreamerv3_results_std = (0.0, 0.0, 0.0, 0.0)

    evaluation_vals = {
        "TD3": tuple(td3_results),
        "PPO": tuple(ppo_results),
        "PPO-LSTM": tuple(ppo_lstm_results),
        "DreamerV3": tuple(dreamerv3_results),
    }
    errors = [
        tuple(td3_results_std),
        tuple(ppo_results_std),
        tuple(ppo_lstm_results_std),
        tuple(dreamerv3_results_std),
    ]

    plot_bar_chart(x_labels, evaluation_vals, save_name, title, errors=errors)


def plot_bar_chart(
    x_labels, evaluation_vals, bar_chart_name, bar_chart_title, errors=None
):
    labels = x_labels

    # errors = [tuple(mavvid_error), tuple(dvb_error), tuple(anti_uav_error)]

    x = np.arange(len(labels))  # the label locations
    bar_width = 0.2

    # for MDPI single column use figsize (10,5)
    # fig, ax = plt.subplots(layout='constrained', figsize=(10, 5), dpi=600)
    # for IEEE double column use figsize (6,3)
    # fig, ax = plt.subplots(layout='constrained', figsize=(6, 3), dpi=600)
    # for springer nature single column use figsize (10,5)
    fig, ax = plt.subplots(layout="constrained", figsize=(9, 3), dpi=600)

    plt.grid(True)
    plt.grid(visible=False, axis="x")
    plt.grid(visible=True, which="major", axis="y", linestyle="-")
    plt.grid(
        visible=True, which="minor", axis="y", linestyle="-", linewidth="0.5", alpha=0.5
    )

    ax.grid(True)
    # ax.minorticks_on()
    ax.set_axisbelow(True)

    # BAR_COLORS = ["#1B4966", "#83C2EA", "#3CA5E6", "#6CD5A1"]
    # BAR_COLORS = ["#42B3FF", "#4942FF", "#4378FF", "#42EFFF"]
    i = 0
    for attribute, measurement in evaluation_vals.items():
        offset = bar_width * i
        # print(f"{measurement=}, {errors[i]=}")
        # measurement = tuple([100*x for x in measurement]) #convert to percentage
        # measurement = tuple([100*x for x in measurement]) #convert to percentage
        # errors[i] = tuple([100*x for x in errors[i]]) #convert to percentage
        # plot bars
        rects = ax.bar(
            x + offset, measurement, bar_width, label=attribute, color=BAR_COLORS[i]
        )
        # plot error bars
        if errors is not None:
            ax.errorbar(
                x + offset,
                measurement,
                yerr=errors[i],
                fmt="none",
                ecolor="black",
                capsize=3,
            )
        ax.bar_label(
            rects,
            padding=-14,
            fmt="%.1f",
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
        )
        i += 1

    # Add labels, title and custom x-axis tick labels
    ax.set_ylabel("$\mathregular{Success Rate (\%)}$")
    ax.set_xlabel("$\mathregular{Noise\ Area\ Size}$")
    ax.set_title(bar_chart_title)
    if len(evaluation_vals) == 2:
        ax.set_xticks((x + bar_width * 0.5), labels=labels)
    elif len(evaluation_vals) == 4:
        ax.set_xticks((x + bar_width * 1.5), labels=labels)
    # ax.legend(loc="upper right", ncols=4)
    ax.legend(loc="upper right")
    # ax.set_ylim(0, 1)
    # ax.set_yticks(np.arange(-1.1, 1.1, 0.2))
    # ax.set_ylim(-1.0, 1.0)
    ax.set_yticks(np.arange(0, 100, 10))
    ax.set_ylim(0, 100)
    # ax.set_yticks(np.arange(0, 101, 10))
    # plt.show()
    plt.savefig(f"figures_journal/bar_charts/{bar_chart_name}.pdf", dpi=600)


def plot_line_chart(
    x_labels, evaluation_vals, line_chart_name, bar_chart_title, errors=None
):
    labels = x_labels

    # errors = [tuple(mavvid_error), tuple(dvb_error), tuple(anti_uav_error)]

    x = np.arange(len(labels))  # the label locations
    bar_width = 0.2

    # for MDPI single column use figsize (10,5)
    # fig, ax = plt.subplots(layout='constrained', figsize=(10, 5), dpi=600)
    # for IEEE double column use figsize (6,3)
    # fig, ax = plt.subplots(layout='constrained', figsize=(6, 3), dpi=600)
    # for springer nature single column use figsize (10,5)
    fig, ax = plt.subplots(layout="constrained", figsize=(9, 3), dpi=600)

    plt.grid(True)
    plt.grid(visible=False, axis="x")
    plt.grid(visible=True, which="major", axis="y", linestyle="-")
    plt.grid(
        visible=True, which="minor", axis="y", linestyle="-", linewidth="0.5", alpha=0.5
    )

    ax.grid(True)
    # ax.minorticks_on()
    ax.set_axisbelow(True)

    # BAR_COLORS = ["#1B4966", "#83C2EA", "#3CA5E6", "#6CD5A1"]
    # BAR_COLORS = ["#42B3FF", "#4942FF", "#4378FF", "#42EFFF"]
    MARKER = ["x-", "o-", "+-", "v-"]
    i = 0
    for attribute, measurement in evaluation_vals.items():
        offset = bar_width * i
        # print(f"{measurement=}, {errors[i]=}")
        # measurement = tuple([100*x for x in measurement]) #convert to percentage
        # measurement = tuple([100*x for x in measurement]) #convert to percentage
        # errors[i] = tuple([100*x for x in errors[i]]) #convert to percentage
        # plot bars

        if errors is not None:
            print(f"{x=}")
            print(f"{measurement=}")
            ax.errorbar(
                x, measurement, yerr=errors[i], label=attribute, color=BAR_COLORS[i]
            )
        else:
            ax.plot(x, measurement, MARKER[i], label=attribute, color=BAR_COLORS[i])
        # rects = ax.bar(
        #     x + offset, measurement, bar_width, label=attribute, color=BAR_COLORS[i]
        # )
        # plot error bars
        # if errors is not None:
        #     ax.errorbar(
        #         x + offset,
        #         measurement,
        #         yerr=errors[i],
        #         fmt="none",
        #         ecolor="black",
        #         capsize=3,
        #     )
        # ax.bar_label(
        #     rects,
        #     padding=-14,
        #     fmt="%.1f",
        #     color="black",
        #     bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
        # )
        i += 1

    # Add labels, title and custom x-axis tick labels
    ax.set_ylabel("$\mathregular{Success Rate (\%)}$")
    ax.set_xlabel("$\mathregular{Evaluation\ Noise\ Area\ Size}$")
    ax.set_title(bar_chart_title)
    ax.set_xticks((x), labels=labels)
    ax.legend(loc="upper right", ncols=4)
    ax.legend(title="Training Regime")

    # ax.set_ylim(0, 1)
    # ax.set_yticks(np.arange(-1.1, 1.1, 0.2))
    # ax.set_ylim(-1.0, 1.0)
    ax.set_yticks(np.arange(0, 100, 10))
    ax.set_ylim(0, 100)
    # ax.set_yticks(np.arange(0, 101, 10))
    # plt.show()
    plt.savefig(f"figures_journal/line_charts/{line_chart_name}.pdf", dpi=600)


def plot_graphs_original():
    path1 = "/home/leo/cranfield-navigation-gym/log_dir/TD3_20240806_120239_lr_exploration_0_003_dt_0_05/tensorboard/TD3_1/events.out.tfevents.1722942187.leonardo.343648.0"
    path2 = "/home/leo/cranfield-navigation-gym/log_dir/TD3_20240805_135318_lr_exploration_0_0003_dt_0_05/tensorboard/TD3_1/events.out.tfevents.1722862432.leonardo.1779531.0"
    path3 = "/home/leo/cranfield-navigation-gym/log_dir/TD3_20240806_172644_lr_exploration_0_00003_dt_0_05/tensorboard/TD3_1/events.out.tfevents.1722961632.leonardo.2261636.0"
    # path1 = path2

    paths = [path1, path2, path3]

    plot_names = []

    plot_names.append("lr=0.003")
    plot_names.append("lr=0.0003")
    plot_names.append("lr=0.00003")

    plot_for_scenario(
        paths, plot_names, "td3_lr_discovery", "TD3 Learning Rate Discovery (Lidar)"
    )

    path4 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20240807_165712_lr_exploration_0_03/tensorboard/PPO_1/events.out.tfevents.1723046258.leonardo.2384018.0"
    path5 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20240806_221447_lr_exploration_0_003/tensorboard/PPO_1/events.out.tfevents.1722978912.leonardo.845087.0"
    path6 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20240807_025624_lr_exploration_0_0003/tensorboard/PPO_1/events.out.tfevents.1722995810.leonardo.1058518.0"
    path7 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20240807_074006_lr_exploration_0_00003/tensorboard/PPO_1/events.out.tfevents.1723012831.leonardo.1506434.0"
    # path1 = path2

    paths = [path4, path5, path6, path7]

    plot_names = []

    plot_names.append("lr=0.03")
    plot_names.append("lr=0.003")
    plot_names.append("lr=0.0003")
    plot_names.append("lr=0.00003")

    plot_for_scenario(
        paths, plot_names, "ppo_lr_discovery", "PPO Learning Rate Discovery (Lidar)"
    )
    # plot_for_scenario(paths, plot_names, "ppo_lr_discovery", "PPO Learning Rate Discovery")

    # --------------------------------------------------PPO------------------------------
    path8 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20240814_041650_camera_noise/tensorboard/PPO_1/events.out.tfevents.1723605435.leonardo.630168.0"
    path9 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20240814_085811_camera_noise/tensorboard/PPO_1/events.out.tfevents.1723622316.leonardo.706207.0"
    path10 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20240814_135612_camera_noise/tensorboard/PPO_1/events.out.tfevents.1723640196.leonardo.2544419.0"
    path11 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20240814_183125_camera_noise/tensorboard/PPO_1/events.out.tfevents.1723656710.leonardo.2171948.0"
    # path1 = path2

    paths_ppo_cam = [path8, path9, path10, path11]

    plot_names_ppo_cam = []

    plot_names_ppo_cam.append("lr=0.03")
    plot_names_ppo_cam.append("lr=0.003")
    plot_names_ppo_cam.append("lr=0.0003")
    plot_names_ppo_cam.append("lr=0.00003")

    plot_for_scenario(
        paths_ppo_cam,
        plot_names_ppo_cam,
        "ppo_cam_lr_discovery",
        "PPO Learning Rate Discovery (Camera)",
    )
    # plot_for_scenario(paths, plot_names, "ppo_lr_discovery", "PPO Learning Rate Discovery")

    path12 = "/home/leo/cranfield-navigation-gym/log_dir/TD3_20240814_231702_camera_noise/tensorboard/TD3_1/events.out.tfevents.1723673848.leonardo.4027355.0"
    path13 = "/home/leo/cranfield-navigation-gym/log_dir/TD3_20240815_085709_camera_noise/tensorboard/TD3_1/events.out.tfevents.1723708655.leonardo.206292.0"
    path14 = "/home/leo/cranfield-navigation-gym/log_dir/TD3_20240815_190807_camera_noise/tensorboard/TD3_1/events.out.tfevents.1723745313.leonardo.3577949.0"
    path15 = "/home/leo/cranfield-navigation-gym/log_dir/TD3_20240816_051346_camera_noise/tensorboard/TD3_1/events.out.tfevents.1723781652.leonardo.3747245.0"
    # path1 = path2

    paths_td3_cam = [path12, path13, path14, path15]

    plot_names_td3_cam = []

    plot_names_td3_cam.append("lr=0.03")
    plot_names_td3_cam.append("lr=0.003")
    plot_names_td3_cam.append("lr=0.0003")
    plot_names_td3_cam.append("lr=0.00003")

    plot_for_scenario(
        paths_td3_cam,
        plot_names_td3_cam,
        "td3_cam_lr_discovery",
        "TD3 Learning Rate Discovery (Camera)",
    )

    # NB: These are lidar
    path16 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_LSTM_20240816_152931_camera_noise/tensorboard/RecurrentPPO_1/events.out.tfevents.1723818596.leonardo.1783785.0"
    path17 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_LSTM_20240816_200819_camera_noise/tensorboard/RecurrentPPO_1/events.out.tfevents.1723835324.leonardo.1470717.0"
    path18 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_LSTM_20240817_012143_camera_noise/tensorboard/RecurrentPPO_1/events.out.tfevents.1723854127.leonardo.1086038.0"
    path19 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_LSTM_20240817_055501_camera_noise/tensorboard/RecurrentPPO_1/events.out.tfevents.1723870525.leonardo.1702066.0"
    # path1 = path2

    paths_ppolstm_lid = [path16, path17, path18, path19]

    plot_names_ppolstm_lid = []

    plot_names_ppolstm_lid.append("lr=0.03")
    plot_names_ppolstm_lid.append("lr=0.003")
    plot_names_ppolstm_lid.append("lr=0.0003")
    plot_names_ppolstm_lid.append("lr=0.00003")

    plot_for_scenario(
        paths_ppolstm_lid,
        plot_names_ppolstm_lid,
        "ppolstm_lid_lr_discovery",
        "PPO LSTM Learning Rate Discovery (Lidar)",
    )

    # NB: These are trained on static goal
    path20 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_LSTM_20240823_203729_ppo_lstm_lr_exploration_0_03_static_goal/tensorboard/RecurrentPPO_1/events.out.tfevents.1724441874.leonardo.3497861.0"
    path21 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_LSTM_20240824_015342_ppo_lstm_lr_exploration_0_003_static_goal/tensorboard/RecurrentPPO_1/events.out.tfevents.1724460847.leonardo.2232167.0"
    path22 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_LSTM_20240822_152735_ppo_lstm_lr_exploration_0_0003_static_goal/tensorboard/RecurrentPPO_1/events.out.tfevents.1724336883.leonardo.2338894.0"
    path23 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_LSTM_20240824_065538_ppo_lstm_lr_exploration_0_00003_static_goal/tensorboard/RecurrentPPO_1/events.out.tfevents.1724478964.leonardo.3859484.0"

    paths_ppolstm_cam = [path20, path21, path22, path23]

    plot_names_ppolstm_cam = []

    plot_names_ppolstm_cam.append("lr=0.03")
    plot_names_ppolstm_cam.append("lr=0.003")
    plot_names_ppolstm_cam.append("lr=0.0003")
    plot_names_ppolstm_cam.append("lr=0.00003")

    plot_for_scenario(
        paths_ppolstm_cam,
        plot_names_ppolstm_cam,
        "ppolstm_cam_lr_discovery",
        "PPO LSTM Learning Rate Discovery (Camera)",
    )

    # Lidar vs cam random, vs cam static
    # NB: These are trained on static goal
    path24 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20240809_163032_PPO_Clean_500k/tensorboard/PPO_1/events.out.tfevents.1723217458.leonardo.3132264.0"
    path25 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/tensorboard/PPO_1/events.out.tfevents.1724355712.leonardo.3532731.0"
    path26 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20240820_055031_camera_500k/tensorboard/PPO_1/events.out.tfevents.1724129456.leonardo.1674394.0"
    # path23 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_LSTM_20240824_065538_ppo_lstm_lr_exploration_0_00003_static_goal/tensorboard/RecurrentPPO_1/events.out.tfevents.1724478964.leonardo.3859484.0"

    paths_modality_study = [path24, path25, path26]

    plot_names_modality_study = []

    plot_names_modality_study.append("Lidar")
    plot_names_modality_study.append("Camera, Static Goal")
    plot_names_modality_study.append("Camera, Random Goal")
    # plot_names_ppolstm_cam.append("lr=0.00003")

    plot_for_scenario(
        paths_modality_study,
        plot_names_modality_study,
        "modality_study",
        "PPO Modality Study",
    )

    path27 = "/home/leo/cranfield-navigation-gym/log_dir/TD3_20240903_132851_camera_lr_study_0_003/tensorboard/TD3_1/events.out.tfevents.1725366559.leonardo.1502080.0"
    path28 = "/home/leo/cranfield-navigation-gym/log_dir/TD3_20240903_190950_camera_lr_study_0_0003/tensorboard/TD3_1/events.out.tfevents.1725387017.leonardo.1548656.0"
    path29 = "/home/leo/cranfield-navigation-gym/log_dir/TD3_20240904_005352_camera_lr_study_0_00003/tensorboard/TD3_1/events.out.tfevents.1725407659.leonardo.3434922.0"
    # path23 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_LSTM_20240824_065538_ppo_lstm_lr_exploration_0_00003_static_goal/tensorboard/RecurrentPPO_1/events.out.tfevents.1724478964.leonardo.3859484.0"

    paths_td3_cam_study = [path27, path28, path29]

    plot_names_td3_cam = []

    plot_names_td3_cam.append("lr=0.003")
    plot_names_td3_cam.append("lr=0.0003")
    plot_names_td3_cam.append("lr=0.00003")

    plot_for_scenario(
        paths_td3_cam_study,
        plot_names_td3_cam,
        "td3_cam_lr_discovery",
        "TD3 Learning Rate Discovery (Camera)",
    )

    path30 = "/home/leo/cranfield-navigation-gym/log_dir/TD3_20240904_062038_camera_static_goal_noise_500k/tensorboard/TD3_1/events.out.tfevents.1725427265.leonardo.1243211.0"
    path31 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/tensorboard/PPO_1/events.out.tfevents.1724355712.leonardo.3532731.0"
    path32 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_LSTM_20240828_120510_camera_static_goal_noise_0/tensorboard/RecurrentPPO_1/events.out.tfevents.1724843140.leonardo.2266479.0"
    path32_5 = "/home/leo/cranfield-navigation-gym/log_dir/dreamer_runs/camera0x0_static/events.out.tfevents.1727476805.inigo.1609120.0.v2"
    # path33 = "/home/leo/dreamerv3/logdir/20240803T230848/events.out.tfevents.1722723077.inigo.3944505.0.v2"
    path33 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20240820_055031_camera_500k/tensorboard/PPO_1/events.out.tfevents.1724129456.leonardo.1674394.0"

    path34 = "/home/leo/cranfield-navigation-gym/log_dir/dreamer_runs/camera0x0/events.out.tfevents.1722723077.inigo.3944505.0.v2"

    paths_algo_cam_study = [path30, path31, path32, path32_5, path33, path34]

    plot_names_algo_cam_study = [
        "TD3",
        "PPO",
        "PPO-LSTM",
        "DreamerV3",
        "PPO (Random Goal)",
        "DreamerV3 (Random Goal)",
    ]

    # plot_names_algo_cam_study.append("Lidar")
    # plot_names_algo_cam_study.append("Camera, Static Goal")
    # plot_names_algo_cam_study.append("Camera, Random Goal")
    # plot_names_ppolstm_cam.append("lr=0.00003")

    plot_algo_scenario(
        paths_algo_cam_study,
        plot_names_algo_cam_study,
        "algo_study",
        "Algorithm Training Study",
    )

    # camer_noise
    plot_noise_graph_camera_0x0(
        "camera_noise_0x0",
        "Evaluating Different Camera Observation Policies Trained on (0x0) Noise Environment",
    )
    plot_noise_graph_camera_3x3(
        "camera_noise_3x3",
        "Evaluating Different Camera Observation Policies Trained on (3x3) Noise Environment",
    )
    plot_noise_graph_camera_0x0_no_obstacles(
        "camera_noise_0x0_no_obstacles",
        "Evaluating Different Camera Observation Policies Trained on the Simplified (0x0) Noise Environment",
    )
    plot_noise_graph_camera_3x3_no_obstacles(
        "camera_noise_3x3_no_obstacles",
        "Evaluating Different Camera Observation Policies Trained on the Simplified (3x3) Noise Environment",
    )

    # plot_noise_graph_camera_temp("camera_noise", "Camera Noise")
    # plot_noise_graph_camera_violin("camera_noise_violin", "Camera Noise")
    # plot_noise_graph_lidar("lidar_noise", "Lidar Noise")
    plot_noise_graph_lidar_0x0(
        "lidar_noise_0x0",
        "Evaluating Different Lidar Observation Policies Trained on (0x0) Noise Environment",
    )
    plot_noise_graph_lidar_3x3(
        "lidar_noise_3x3",
        "Evaluating Different Lidar Observation Policies Trained on (3x3) Noise Environment",
    )

    plot_noise_graph_camera_ppo_line(
        "camera_noise_configurations_ppo",
        "Evaluation Performance on Different Camera Noise Scenarios Across Different Training Regimes (PPO)",
    )

    plot_noise_graph_camera_ppo_line_no_obstacle_map(
        "camera_noise_configurations_ppo_no_obstacle_map",
        "Evaluation Performance on Different Camera Noise Scenarios Across Different Training Regimes (PPO, No Obstacle Map)",
    )

    plot_noise_graph_camera_ppo_lstm_no_obstacle_map(
        "camera_noise_configurations_ppo_lstm_no_obstacle_map",
        "Evaluation Performance on Different Camera Noise Scenarios Across Different Training Regimes (PPO LSTM, No Obstacle Map)",
    )

    plot_noise_graph_camera_td3_no_obstacle_map(
        "camera_noise_configurations_td3_no_obstacle_map",
        "Evaluation Performance on Different Camera Noise Scenarios Across Different Training Regimes (TD3, No Obstacle Map)",
    )

    plot_noise_graph_camera_ppo_line_error(
        "camera_noise_ppo_errors",
        "Evaluation Performance on Different Camera Noise Scenarios Across Different Training Regimes (PPO)",
    )

    plot_noise_graph_camera_ppo_lstm_line(
        "camera_noise_configurations_ppo_lstm",
        "Evaluation Performance on Different Camera Noise Scenarios Across Different Training Regimes (PPO LSTM)",
    )

    plot_noise_graph_camera_td3_line(
        "camera_noise_configurations_td3",
        "Evaluation Performance on Different Camera Noise Scenarios Across Different Training Regimes (TD3)",
    )

    # lidar line plots
    plot_noise_graph_lidar_td3_line_error(
        "lidar_noise_configurations_td3",
        "Evaluation Performance on Different Lidar Noise Scenarios Across Different Training Regimes (TD3)",
    )

    plot_noise_graph_lidar_ppo_line_error(
        "lidar_noise_configurations_ppo",
        "Evaluation Performance on Different Lidar Noise Scenarios Across Different Training Regimes (PPO)",
    )

    save_path = "./figures_journal/"
    import os

    print(f"Saved figures to path {os.path.abspath(save_path)}")
    return


def plot_graphs_march_25_pr_update():
    # 28_03 LR plots
    # PPO Camera Random Goal
    path35 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20250319_180851_PPO_random_goal_continuous_act_lr_framestack/tensorboard/PPO_1/events.out.tfevents.1742407740.leonardo.2176051.0"
    path36 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20250320_003327_PPO_random_goal_continuous_act_lr_framestack/tensorboard/PPO_1/events.out.tfevents.1742430816.leonardo.535425.0"
    path37 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20250320_070018_PPO_random_goal_continuous_act_lr_framestack/tensorboard/PPO_1/events.out.tfevents.1742454027.leonardo.3071475.0"
    path38 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20250320_132432_PPO_random_goal_continuous_act_lr_framestack/tensorboard/PPO_1/events.out.tfevents.1742477082.leonardo.1397226.0"
    # path1 = path2

    paths_ppo_cam_randgoal = [path35, path36, path37, path38]

    plot_names_ppo_cam_randgoal = []

    plot_names_ppo_cam_randgoal.append("lr=0.000003, framestack")
    plot_names_ppo_cam_randgoal.append("lr=0.00003, framestack")
    plot_names_ppo_cam_randgoal.append("lr=0.0003, framestack")
    plot_names_ppo_cam_randgoal.append("lr=0.003, framestack")

    plot_for_scenario(
        paths_ppo_cam_randgoal,
        plot_names_ppo_cam_randgoal,
        "ppo_lr_discovery_randomgoal",
        "PPO Learning Rate Discovery (Camera, Framestack)",
    )

    path30 = "/home/leo/cranfield-navigation-gym-dev/log_dir/TD3_20240904_062038_camera_static_goal_noise_500k/tensorboard/TD3_1/events.out.tfevents.1725427265.leonardo.1243211.0"
    path31 = "/home/leo/cranfield-navigation-gym-dev/log_dir/PPO_20240822_204128_camera_static_goal/tensorboard/PPO_1/events.out.tfevents.1724355712.leonardo.3532731.0"
    path32 = "/home/leo/cranfield-navigation-gym-dev/log_dir/PPO_LSTM_20240828_120510_camera_static_goal_noise_0/tensorboard/RecurrentPPO_1/events.out.tfevents.1724843140.leonardo.2266479.0"
    path32_5 = "/home/leo/cranfield-navigation-gym-dev/log_dir/dreamer_runs/camera0x0_static/events.out.tfevents.1727476805.inigo.1609120.0.v2"
    # path33 = "/home/leo/dreamerv3/logdir/20240803T230848/events.out.tfevents.1722723077.inigo.3944505.0.v2"

    paths_algo_cam_study = [path30, path31, path32, path32_5]

    plot_names_algo_cam_study = [
        "TD3",
        "PPO",
        "PPO-LSTM",
        "DreamerV3",
    ]

    plot_algo_scenario2(
        paths_algo_cam_study,
        plot_names_algo_cam_study,
        "algo_study",
        "Algorithm Training Study (Camera, Static Goal)",
    )

    path33_td3 = "/home/leo/cranfield-navigation-gym/log_dir/td3_29_03_25/TD3_20250329_120700_random_goal_td3_cont_act_orig_td3_params/tensorboard/TD3_1/events.out.tfevents.1743250031.inigo.6275.0"
    path34_ppo = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20250326_153225_PPO_random_goal_continuous_act_lr/tensorboard/PPO_1/events.out.tfevents.1743003154.leonardo.3772545.0"
    path35_ppo_lstm = "/home/leo/cranfield-navigation-gym/log_dir/PPO_LSTM_20250330_151337_PPO_LSTM_random_goal_continuous_act_lr0_00003/tensorboard/RecurrentPPO_1/events.out.tfevents.1743344026.leonardo.1391421.0"
    path36_dreamer = "/home/leo/cranfield-navigation-gym/log_dir/dreamerv3_29_03_25/20250330T165535/events.out.tfevents.1743350284.inigo.4178754.0.v2"

    paths_algo_cam_random_study = [
        path33_td3,
        path34_ppo,
        path35_ppo_lstm,
        path36_dreamer,
    ]

    plot_names_algo_cam_random_study = [
        "TD3",
        "PPO",
        "PPO-LSTM",
        "DreamerV3",
    ]

    plot_algo_scenario3(
        paths_algo_cam_random_study,
        plot_names_algo_cam_random_study,
        "algo_study_random_cam",
        "Algorithm Training Study (Camera, Random Goal)",
    )

    plot_algo_scenario4_wall_time(
        paths_algo_cam_random_study,
        plot_names_algo_cam_random_study,
        "algo_study_random_cam_wall_time",
        "Algorithm Training Study (Camera, Random Goal) - Wall Time",
    )

    # 0x0 vs 3x3 training
    path33_td3_0x0 = "/home/leo/cranfield-navigation-gym/log_dir/td3_29_03_25/TD3_20250329_120700_random_goal_td3_cont_act_orig_td3_params/tensorboard/TD3_1/events.out.tfevents.1743250031.inigo.6275.0"
    path33_td3_3x3 = "/home/leo/cranfield-navigation-gym/log_dir/td3_29_03_25/TD3_20250330_023606_random_goal_td3_cont_act_orig_td3_params_noise/tensorboard/TD3_1/events.out.tfevents.1743298577.inigo.4176280.0"
    path34_ppo_0x0 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20250326_153225_PPO_random_goal_continuous_act_lr/tensorboard/PPO_1/events.out.tfevents.1743003154.leonardo.3772545.0"
    path34_ppo_3x3 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20250326_181754_PPO_random_goal_continuous_act_noise/tensorboard/PPO_1/events.out.tfevents.1743013085.leonardo.26286.0"
    path35_ppo_lstm_0x0 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_LSTM_20250330_151337_PPO_LSTM_random_goal_continuous_act_lr0_00003/tensorboard/RecurrentPPO_1/events.out.tfevents.1743344026.leonardo.1391421.0"
    path35_ppo_lstm_3x3 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_LSTM_20250331_104411_PPO_LSTM_random_goal_continuous_act_lr0_00003_noise3/tensorboard/RecurrentPPO_1/events.out.tfevents.1743414260.leonardo.1406011.0"
    path36_dreamer_0x0 = "/home/leo/cranfield-navigation-gym/log_dir/dreamerv3_29_03_25/20250330T165535/events.out.tfevents.1743350284.inigo.4178754.0.v2"
    path36_dreamer_3x3 = "/home/leo/cranfield-navigation-gym/log_dir/dreamerv3_29_03_25/20250330T231707/events.out.tfevents.1743373175.inigo.160191.0.v2"

    paths_algo_cam_random_noise_study = [
        path33_td3_0x0,
        path33_td3_3x3,
        path34_ppo_0x0,
        path34_ppo_3x3,
        path35_ppo_lstm_0x0,
        path35_ppo_lstm_3x3,
        path36_dreamer_0x0,
        path36_dreamer_3x3,
    ]

    plot_names_algo_cam_random_noise_study = [
        "TD3 0x0",
        "TD3 3x3",
        "PPO 0x0",
        "PPO 3x3",
        "PPO-LSTM 0x0",
        "PPO-LSTM 3x3",
        "DreamerV3 0x0",
        "DreamerV3 3x3",
    ]

    plot_algo_scenario5(
        paths_algo_cam_random_noise_study,
        plot_names_algo_cam_random_noise_study,
        "algo_study_random_cam_noise",
        "Algorithm Training Study (Camera, Random Goal), Vanilla vs Adversarial Training",
    )

    # "PPO (Random Goal)",
    # "DreamerV3 (Random Goal)",

    # path1 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20250325_201031_PPO_random_goal_continuous_act_lr/tensorboard/PPO_1/events.out.tfevents.1742933440.leonardo.299138.0"
    # path2 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20250326_104233_PPO_random_goal_continuous_act_lr/tensorboard/PPO_1/events.out.tfevents.1742985764.leonardo.2923233.0"
    # path3 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20250326_153225_PPO_random_goal_continuous_act_lr/tensorboard/PPO_1/events.out.tfevents.1743003154.leonardo.3772545.0"

    # paths_ppo_cam_randgoal = [path1, path2, path3]

    # plot_names_ppo_cam_randgoal = []

    # plot_names_ppo_cam_randgoal.append("lr=0.00003, 1")
    # plot_names_ppo_cam_randgoal.append("lr=0.00003, 2")
    # plot_names_ppo_cam_randgoal.append("lr=0.00003, 3")

    # plot_for_scenario(
    #     paths_ppo_cam_randgoal,
    #     plot_names_ppo_cam_randgoal,
    #     "ppo_lr_discovery_repeat",
    #     "PPO Learning Rate Discovery (Camera)",
    # )

    # PPO random goal lr
    path1 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20250318_083605_PPO_static_goal_continuous_act/tensorboard/PPO_1/events.out.tfevents.1742286974.leonardo.3637658.0"
    path2 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20250318_034726_PPO_static_goal_continuous_act/tensorboard/PPO_1/events.out.tfevents.1742269655.leonardo.2183248.0"
    path3 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20250317_225728_PPO_static_goal_continuous_act/tensorboard/PPO_1/events.out.tfevents.1742252257.leonardo.729679.0"
    path4 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20250317_180834_PPO_static_goal_continuous_act/tensorboard/PPO_1/events.out.tfevents.1742234924.leonardo.3480377.0"
    # path5 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20250317_142422_PPO_static_goal_continuous_act/tensorboard/PPO_1/events.out.tfevents.1742221471.leonardo.2416606.0"

    # path6 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20250318_132204_PPO_static_goal_continuous_act/tensorboard/PPO_1/events.out.tfevents.1742304133.leonardo.850475.0" #PPO lr 3e-7

    # path1 = path2

    paths_ppo_cam_randgoal = [path1, path2, path3, path4]

    plot_names_ppo_cam_randgoal = []

    plot_names_ppo_cam_randgoal.append("lr=0.003")
    plot_names_ppo_cam_randgoal.append("lr=0.0003")
    plot_names_ppo_cam_randgoal.append("lr=0.000003")
    plot_names_ppo_cam_randgoal.append("lr=0.00003")
    # plot_names_ppo_cam_randgoal.append("lr=0.00003")
    # plot_names_ppo_cam_randgoal.append("lr=0.0000003")

    # plot_names_ppo_cam_randgoal.append("lr=0.00003")

    plot_for_scenario(
        paths_ppo_cam_randgoal,
        plot_names_ppo_cam_randgoal,
        "ppo_lr_discovery",
        "PPO Learning Rate Discovery (Camera)",
    )

    plot_noise_graph_camera_0x0_randomgoal(
        "camera_noise_0x0_randomgoal",
        "Evaluating Different Camera Observation Policies Trained on (0x0) Noise Environment (Random Goal)",
    )
    plot_noise_graph_camera_3x3_randomgoal(
        "camera_noise_3x3_randomgoal",
        "Evaluating Different Camera Observation Policies Trained on (3x3) Noise Environment (Random Goal)",
    )

    path24 = "/home/leo/cranfield-navigation-gym-dev/log_dir/PPO_20240809_163032_PPO_Clean_500k/tensorboard/PPO_1/events.out.tfevents.1723217458.leonardo.3132264.0"
    path25 = "/home/leo/cranfield-navigation-gym-dev/log_dir/PPO_20240822_204128_camera_static_goal/tensorboard/PPO_1/events.out.tfevents.1724355712.leonardo.3532731.0"
    path26 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_20250326_153225_PPO_random_goal_continuous_act_lr/tensorboard/PPO_1/events.out.tfevents.1743003154.leonardo.3772545.0"
    # path23 = "/home/leo/cranfield-navigation-gym/log_dir/PPO_LSTM_20240824_065538_ppo_lstm_lr_exploration_0_00003_static_goal/tensorboard/RecurrentPPO_1/events.out.tfevents.1724478964.leonardo.3859484.0"

    paths_modality_study = [path24, path25, path26]

    plot_names_modality_study = []

    plot_names_modality_study.append("Lidar")
    plot_names_modality_study.append("Camera, Static Goal")
    plot_names_modality_study.append("Camera, Random Goal")
    # plot_names_ppolstm_cam.append("lr=0.00003")

    plot_for_scenario(
        paths_modality_study,
        plot_names_modality_study,
        "modality_study",
        "PPO Modality Study",
    )


if __name__ == "__main__":
    # plot_graphs_original()

    plot_graphs_march_25_pr_update()
