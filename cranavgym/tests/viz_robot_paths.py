import matplotlib
import seaborn as sns
import seaborn.objects as so
import pandas as pd

matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Bitstream Vera Sans"
matplotlib.rcParams["mathtext.it"] = "Bitstream Vera Sans:italic"
matplotlib.rcParams["mathtext.bf"] = "Bitstream Vera Sans:bold"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def main(df_path, bk_path, save_name, title):
    df = load_df(df_path)
    xy_pos = get_xy_positions(df, 0)

    # plot_path_with_value(df, bk_path, save_name)
    # plot_path(df, bk_path, save_name, title)

    plot_tsne(df)

    return


def load_df(path):
    df = pd.read_pickle(path)
    return df


def get_xy_positions(df, episode):
    x_pos = df.loc[df["episode"] == episode, "x_position"]
    y_pos = df.loc[df["episode"] == episode, "y_position"]

    return (x_pos, y_pos)


def plot_path_with_value(df, bk_path, save_name):
    if bk_path is not None:
        bg_image = plt.imread(bk_path)

    # Create figure and axes
    fig, ax = plt.subplots(layout="constrained", figsize=(6, 6), dpi=600)

    # Display the background image
    if bk_path is not None:
        ax.imshow(bg_image, extent=[-5.5, 5.5, -5.5, 5.5], aspect="equal")

    df_ep_0 = df.loc[df["episode"] == 1]
    # sns.jointplot(data=df, x="x_position", y="y_position", hue="episode", xlim=(-5, 5), ylim=(-5, 5))
    sns.scatterplot(data=df[0:500], x="x_position", y="y_position", hue="value", ax=ax)

    # for vectors
    # import numpy as np
    # M = np.hypot(df['x_velocity'], df['y_velocity'])
    # Q = ax.quiver(df['x_position'], df['y_position'], df['x_velocity'], df['y_velocity'], M, units='x', pivot='tail', width=0.022,
    #            scale=1 / 0.15)
    plt.savefig(f"./figures_journal/value_functions/{save_name}.pdf", dpi=600)
    # plt.show()

    return


def plot_path(df, bk_path, save_name, title):
    if bk_path is not None:
        bg_image = plt.imread(bk_path)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(4, 4), dpi=600)

    # Display the background image
    bounds = 5.5
    if bk_path is not None:
        ax.imshow(bg_image, extent=[-bounds, bounds, -bounds, bounds], aspect="equal")

    df_ep_0 = df.loc[df["episode"] == 1]

    # sns.pointplot(data=df_ep_0, x="x_position", y="y_position")

    # p = so.Plot(df[:500], x="x_position", y="y_position", color="episode")
    # p.add(so.Path())

    # p.add(so.Path(marker="o", pointsize=2, linewidth=.75, fillcolor="w"))
    # p.sow()

    # add noise rectangle
    # noise_bounds = 1.5
    # ax.fill([-noise_bounds,noise_bounds,noise_bounds,-noise_bounds], [-noise_bounds,-noise_bounds,noise_bounds,noise_bounds], 'b', alpha=0.5, zorder=3)

    # sns.jointplot(data=df_ep_0, x="x_position", y="y_position", hue="episode", xlim=(-5, 5), ylim=(-5, 5))
    sns.lineplot(
        data=df,
        x="x_position",
        y="y_position",
        hue="episode",
        sort=False,
        ax=ax,
        markers=True,
        legend=False,
    )
    sns.scatterplot(
        data=(df.loc[df["step_reward"] == -1]),
        x="x_position",
        y="y_position",
        size=1,
        color="red",
        ax=ax,
        zorder=3,
    )
    sns.scatterplot(
        data=(df.loc[df["step_reward"] == 1]),
        x="x_position",
        y="y_position",
        size=1,
        color="green",
        ax=ax,
        zorder=3,
    )

    success_marker = mlines.Line2D(
        [],
        [],
        color="green",
        marker="o",
        linestyle="None",
        markersize=5,
        label=f"{df.loc[df['step_reward'] == 1].count().episode} Success",
    )

    fail_marker = mlines.Line2D(
        [],
        [],
        color="red",
        marker="o",
        linestyle="None",
        markersize=5,
        label=f"{df.loc[df['step_reward'] == -1].count().episode} Crashes",
    )

    plt.legend(loc="lower left", handles=[success_marker, fail_marker])

    plt.title(title)
    plt.show()

    # for vectors
    # import numpy as np
    # M = np.hypot(df['x_velocity'], df['y_velocity'])
    # Q = ax.quiver(df['x_position'], df['y_position'], df['x_velocity'], df['y_velocity'], M, units='x', pivot='tail', width=0.022,
    #            scale=1 / 0.15)
    plt.savefig(f"./figures_journal/paths/{save_name}.pdf", dpi=600)

    # plt.show()


def plot_tsne(df):
    # -------------------------------TSNE------------------------

    # tsne latent reps
    from sklearn.manifold import TSNE
    import numpy as np
    import os

    latent_features = []
    log_dir = "/home/leo/cranfield-navigation-gym/log_dir/evaluation/camera_repeatability/PPO_20250325_185905_PPO_continuous/lf/"
    for i in range(9466):
        lf = np.load(os.path.join(log_dir, f"latent_features{i}.npy"))
        latent_features.append(lf)

    tsne = TSNE(n_components=2)

    # latent_reps_eps_np = np.array(latent_features)
    latent_reps_np = np.array(latent_features)
    print(latent_reps_np.shape)
    tsne_results = tsne.fit_transform(latent_reps_np)
    print(tsne_results.shape)

    print(f"{df}")
    print(f"{df}")

    sns.color_palette("tab10")

    sns.scatterplot(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        hue=df["value"].astype(float),
        palette="viridis",
    )
    plt.show()
    plt.savefig(f"./figures_journal/paths/tsne.pdf", dpi=600)

    return


if __name__ == "__main__":
    # df_path = "~/cranfield-navigation-gym/cranavgym/tests/test.pkl"

    # obstacle_map
    # df_path = "~/cranfield-navigation-gym/log_dir/evaluation/0x0_evaluation/PPO_20240916_140744_default_baseline/evaluation_results_raw.pkl"
    df_path = "~/cranfield-navigation-gym/log_dir/evaluation/camera_repeatability/PPO_20250325_185905_PPO_continuous/evaluation_results_raw.pkl"
    # bk_w/obstacles
    bk_path = (
        "/home/leo/cranfield-navigation-gym/cranavgym/tests/value_function_bkground.png"
    )

    # main(df_path, bk_path, "obstacle_ppo")

    # bk_w/obstacles
    bk_no_obstacles_path = (
        "/home/leo/cranfield-navigation-gym/cranavgym/tests/value_function_bkground.png"
    )

    # value funct plots
    # main("~/cranfield-navigation-gym/log_dir/evaluation/value_function/PPO_20240925_163118_default_baseline_default_map/evaluation_results_raw.pkl", bk_no_obstacles_path, "value_funct_ppo_0x0_eval_0x0")
    # main("~/cranfield-navigation-gym/log_dir/evaluation/value_function/PPO_20240925_163539_default_3x3_default_map/evaluation_results_raw.pkl", bk_no_obstacles_path, "value_funct_ppo_0x0_eval_3x3")
    # main("~/cranfield-navigation-gym/log_dir/evaluation/value_function/PPO_20240925_165242_value_funct_ppo_3x3_default_map_0x0/evaluation_results_raw.pkl", bk_no_obstacles_path, "value_funct_ppo_3x3_eval_0x0")
    # main("~/cranfield-navigation-gym/log_dir/evaluation/value_function/PPO_20240925_165730_value_funct_ppo_3x3_default_map_3x3/evaluation_results_raw.pkl", bk_no_obstacles_path, "value_funct_ppo_3x3_eval_3x3")

    # path plots
    # main("~/cranfield-navigation-gym/log_dir/evaluation/lidar_evals/TD3_20240925_182023_default_baseline_default_map/evaluation_results_raw.pkl", bk_no_obstacles_path, "td3_0x0_eval_0x0_lidar_path", "TD3 Trained on 0x0 Noise, Evaluated on 0x0 Noise")
    # main("~/cranfield-navigation-gym/log_dir/evaluation/lidar_evals/TD3_20240925_183515_default_7x7_default_map/evaluation_results_raw.pkl", bk_no_obstacles_path, "td3_0x0_eval_7x7_lidar_path", "TD3 Trained on 0x0 Noise, Evaluated on 7x7 Noise" )

    # main("~/cranfield-navigation-gym/log_dir/evaluation/lidar_evals/PPO_20240925_192032_default_baseline_default_map/evaluation_results_raw.pkl", bk_no_obstacles_path, "ppo_0x0_eval_0x0_lidar_path", "PPO Trained on 0x0 Noise, Evaluated on 0x0 Noise")
    # main("~/cranfield-navigation-gym/log_dir/evaluation/lidar_evals/PPO_20240925_194202_default_7x7_default_map/evaluation_results_raw.pkl", bk_no_obstacles_path, "ppo_0x0_eval_7x7_lidar_path", "PPO Trained on 0x0 Noise, Evaluated on 7x7 Noise")
    # main("~/cranfield-navigation-gym/log_dir/evaluation/lidar_evals/PPO_20240925_195257_ppo_3x3_default_map_0x0/evaluation_results_raw.pkl", bk_no_obstacles_path, "ppo_3x3_eval_0x0_lidar_path", "PPO Trained on 3x3 Noise, Evaluated on 0x0 Noise")
    # main("~/cranfield-navigation-gym/log_dir/evaluation/lidar_evals/PPO_20240925_200939_ppo_3x3_default_map_7x7/evaluation_results_raw.pkl", bk_no_obstacles_path, "ppo_3x3_eval_7x7_lidar_path", "PPO Trained on 3x3 Noise, Evaluated on 7x7 Noise")

    # camera no obstacle runs

    # main(
    #     "~/cranfield-navigation-gym/log_dir/evaluation/value_function/PPO_20240925_163118_default_baseline_default_map/evaluation_results_raw.pkl",
    #     None,
    #     "ppo_0x0_eval_0x0_camera_path",
    #     "PPO Trained on 0x0 Noise, Evaluated on 0x0 Noise",
    # )
    # # main("~/cranfield-navigation-gym/log_dir/evaluation/value_function/PPO_20240925_163539_default_3x3_default_map/evaluation_results_raw.pkl", None, "ppo_0x0_eval_3x3_camera_path", "PPO Trained on 0x0 Noise, Evaluated on 3x3 Noise")
    # main(
    #     "~/cranfield-navigation-gym/log_dir/evaluation/value_function/PPO_20240925_165242_value_funct_ppo_3x3_default_map_0x0/evaluation_results_raw.pkl",
    #     None,
    #     "ppo_3x3_eval_0x0_camera_path",
    #     "PPO Trained on 3x3 Noise, Evaluated on 0x0 Noise",
    # )

    main(
        "~/cranfield-navigation-gym/log_dir/evaluation/camera_repeatability/PPO_20250325_185905_PPO_continuous/evaluation_results_raw.pkl",
        None,
        "ppo_0x0_eval_0x0_camera_path_Q",
        "PPO Trained on 0x0 Noise, Evaluated on 0x0 Noise, Q",
    )
    # main("~/cranfield-navigation-gym/log_dir/evaluation/value_function/PPO_20240925_165730_value_funct_ppo_3x3_default_map_3x3/evaluation_results_raw.pkl", None, "ppo_3x3_eval_3x3_camera_path", "PPO Trained on 3x3 Noise, Evaluated on 3x3 Noise")
