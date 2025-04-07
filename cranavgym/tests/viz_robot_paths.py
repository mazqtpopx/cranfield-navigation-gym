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
from sklearn.manifold import TSNE
import numpy as np
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def main(run_path, bk_path, save_name, title):
    eval_pkl_name = "evaluation_results_raw.pkl"

    df_path = os.path.join(run_path, eval_pkl_name)
    df = load_df(df_path)
    xy_pos = get_xy_positions(df, 0)

    # plot_path_with_value(df, bk_path, save_name)
    # plot_path(df, bk_path, save_name, title)

    # plot_tsne(df, save_name, run_path)

    plot_tsne_with_click_range(df, save_name, run_path)
    # plot_tsne_with_highlighted_point(df, save_name, run_path)

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
    sns.scatterplot(data=df[0:2000], x="x_position", y="y_position", hue="value", ax=ax)

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
    # plt.show()

    # for vectors
    # import numpy as np
    # M = np.hypot(df['x_velocity'], df['y_velocity'])
    # Q = ax.quiver(df['x_position'], df['y_position'], df['x_velocity'], df['y_velocity'], M, units='x', pivot='tail', width=0.022,
    #            scale=1 / 0.15)
    plt.savefig(f"./figures_journal/paths/{save_name}.pdf", dpi=600)

    # plt.show()


def plot_tsne(df, save_name, run_dir):
    # -------------------------------TSNE------------------------

    # tsne latent reps

    latent_features = []
    lf_dir = os.path.join(run_dir, "lf")

    # number of latent feature files in the directory
    no_of_lfs = len(
        [
            name
            for name in os.listdir(lf_dir)
            if os.path.isfile(os.path.join(lf_dir, name))
        ]
    )

    print(f"{no_of_lfs=}")

    for i in range(no_of_lfs):
        lf = np.load(os.path.join(lf_dir, f"latent_features{i}.npy"))
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
    # plt.savefig(f"./figures_journal/tsne_plots/{save_name}.pdf", dpi=600)

    return


def plot_tsne_with_click(df, save_name, run_dir):
    # -------------------------------TSNE------------------------

    latent_features = []
    lf_dir = os.path.join(run_dir, "lf")

    # Load latent features
    no_of_lfs = len(
        [name for name in os.listdir(lf_dir) if name.startswith("latent_features")]
    )

    print(f"{no_of_lfs=}")

    for i in range(no_of_lfs):
        lf = np.load(os.path.join(lf_dir, f"latent_features{i}.npy"))
        latent_features.append(lf)

    tsne = TSNE(n_components=2)
    latent_reps_np = np.array(latent_features)
    tsne_results = tsne.fit_transform(latent_reps_np)

    # Plot the t-SNE results
    # fig, ax = plt.subplots(layout="constrained", figsize=(12, 12), dpi=600)
    fig, ax = plt.subplots(layout="constrained", figsize=(9, 5), dpi=200)
    # fig, ax = plt.subplots()
    sns.scatterplot(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        hue=df["value"].astype(float),
        palette="viridis",
        ax=ax,
    )

    # Function to show image when clicked
    def on_click(event):
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata

            # Find the closest point
            distances = np.hypot(tsne_results[:, 0] - x, tsne_results[:, 1] - y)
            closest_index = np.argmin(distances)

            # Display image associated with the point
            img_path = os.path.join(
                os.path.join(run_dir, "states"), f"state{closest_index}.png"
            )

            if os.path.exists(img_path):
                print(f"Opening: {img_path}")
                img = plt.imread(img_path)

                # Display image next to the point
                imgbox = OffsetImage(img, zoom=0.3)
                ab = AnnotationBbox(imgbox, (x, y), frameon=False)
                ax.add_artist(ab)
                plt.draw()
            else:
                print(f"Image not found: {img_path}")

    # Connect the click event
    fig.canvas.mpl_connect("button_press_event", on_click)

    # plt.colorbar("viridis")
    # plt.legend(False)

    plt.show()
    # plt.savefig(f"./figures_journal/tsne_plots/{save_name}.pdf", dpi=600)


def plot_tsne_with_click_range(df, save_name, run_dir):
    # -------------------------------TSNE------------------------

    latent_features = []
    lf_dir = os.path.join(run_dir, "lf")

    new_range = 5000

    # Load latent features
    no_of_lfs = len(
        [name for name in os.listdir(lf_dir) if name.startswith("latent_features")]
    )

    print(f"{no_of_lfs=}")

    for i in range(new_range):
        lf = np.load(os.path.join(lf_dir, f"latent_features{i}.npy"))
        latent_features.append(lf)

    tsne = TSNE(n_components=2)
    latent_reps_np = np.array(latent_features)
    tsne_results = tsne.fit_transform(latent_reps_np)

    # Plot the t-SNE results
    # fig, ax = plt.subplots(layout="constrained", figsize=(12, 12), dpi=600)
    fig, ax = plt.subplots(layout="constrained", figsize=(9, 5), dpi=200)
    # fig, ax = plt.subplots()

    df = df[0:new_range]
    sns.scatterplot(
        x=tsne_results[0:new_range, 0],
        y=tsne_results[0:new_range, 1],
        hue=df["value"].astype(float),
        palette="viridis",
        ax=ax,
    )

    # Function to show image when clicked
    def on_click(event):
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata

            # Find the closest point
            distances = np.hypot(tsne_results[:, 0] - x, tsne_results[:, 1] - y)
            closest_index = np.argmin(distances)

            # Display image associated with the point
            img_path = os.path.join(
                os.path.join(run_dir, "states"), f"state{closest_index}.png"
            )

            if os.path.exists(img_path):
                print(f"Opening: {img_path}")
                img = plt.imread(img_path)

                # Display image next to the point
                imgbox = OffsetImage(img, zoom=0.3)
                ab = AnnotationBbox(imgbox, (x, y), frameon=False)
                ax.add_artist(ab)
                plt.draw()
            else:
                print(f"Image not found: {img_path}")

    # Connect the click event
    fig.canvas.mpl_connect("button_press_event", on_click)

    # plt.colorbar("viridis")
    # plt.legend(False)

    plt.show()


def plot_tsne_with_external_images(df, save_name, run_dir):
    # -------------------------------TSNE------------------------

    latent_features = []
    lf_dir = os.path.join(run_dir, "lf")

    # Load latent features
    no_of_lfs = len(
        [name for name in os.listdir(lf_dir) if name.startswith("latent_features")]
    )

    print(f"{no_of_lfs=}")

    for i in range(no_of_lfs):
        lf = np.load(os.path.join(lf_dir, f"latent_features{i}.npy"))
        latent_features.append(lf)

    # Fit t-SNE
    tsne = TSNE(n_components=2)
    latent_reps_np = np.array(latent_features)
    tsne_results = tsne.fit_transform(latent_reps_np)

    # Create figure with space for images on the right
    fig, ax = plt.subplots(figsize=(12, 8))

    # Extra space for images
    plt.subplots_adjust(right=0.75)  # Reserve space on the right for images

    # Plot the t-SNE scatter plot
    sns.scatterplot(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        hue=df["value"].astype(float),
        palette="viridis",
        ax=ax,
    )

    # Store AnnotationBbox references to clear previous images
    img_annotations = []

    def on_click(event):
        nonlocal img_annotations

        if event.inaxes is not None:
            x, y = event.xdata, event.ydata

            # Find the closest point
            distances = np.hypot(tsne_results[:, 0] - x, tsne_results[:, 1] - y)
            closest_index = np.argmin(distances)

            img_path = os.path.join(
                os.path.join(run_dir, "states"), f"state{closest_index}.png"
            )

            if os.path.exists(img_path):
                print(f"Opening: {img_path}")
                img = plt.imread(img_path)

                # Clear previous images
                for ab in img_annotations:
                    ab.remove()
                img_annotations.clear()

                # Display image on the right
                img_x = tsne_results[:, 0].max() + 100.0  # Place image to the right
                img_y = y

                # Add the image
                imgbox = OffsetImage(img, zoom=0.5)
                ab = AnnotationBbox(imgbox, (img_x, img_y), frameon=True)
                img_annotations.append(ax.add_artist(ab))

                # Draw line connecting point to image
                ax.plot(
                    [x, img_x], [y, img_y], color="gray", linestyle="--", linewidth=1
                )

                plt.draw()
            else:
                print(f"Image not found: {img_path}")

    # Connect the click event
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()


def plot_tsne_with_highlighted_point(df, save_name, run_dir):
    # -------------------------------TSNE------------------------

    latent_features = []
    lf_dir = os.path.join(run_dir, "lf")

    # Load latent features
    no_of_lfs = len(
        [name for name in os.listdir(lf_dir) if name.startswith("latent_features")]
    )

    print(f"{no_of_lfs=}")

    for i in range(no_of_lfs):
        if i > 1:
            i = i - 1
        lf = np.load(os.path.join(lf_dir, f"latent_features{i}.npy"))
        latent_features.append(lf)

    # Fit t-SNE
    tsne = TSNE(n_components=2)
    latent_reps_np = np.array(latent_features)
    tsne_results = tsne.fit_transform(latent_reps_np)

    # Create a figure with two axes
    fig, (ax_main, ax_img) = plt.subplots(
        1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [3, 1]}
    )

    # Plot t-SNE in the main axis
    scatter = sns.scatterplot(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        hue=df["value"].astype(float),
        palette="viridis",
        ax=ax_main,
    )

    ax_main.set_title("t-SNE Plot")
    ax_img.set_title("Selected Image")
    ax_img.axis("off")  # Hide the axes for the image panel

    # Store the current highlighted point
    highlighted_point = None

    def on_click(event):
        nonlocal highlighted_point

        if event.inaxes == ax_main:
            x, y = event.xdata, event.ydata

            # Find the closest point
            distances = np.hypot(tsne_results[:, 0] - x, tsne_results[:, 1] - y)
            closest_index = np.argmin(distances)

            if closest_index > 1:
                closest_index = closest_index - 1
            img_path = os.path.join(
                os.path.join(run_dir, "states"), f"state{closest_index}.png"
            )

            if os.path.exists(img_path):
                print(f"Displaying: {img_path}")
                img = plt.imread(img_path)

                # Clear the previous image and display the new one
                ax_img.clear()
                ax_img.imshow(img)
                ax_img.axis("off")
                ax_img.set_title(f"Image {closest_index}")

                # Highlight the clicked point
                if highlighted_point:
                    highlighted_point.remove()  # Remove previous highlight

                # Add new highlight marker
                highlighted_point = ax_main.scatter(
                    tsne_results[closest_index, 0],
                    tsne_results[closest_index, 1],
                    color="red",
                    marker="o",
                    s=150,
                    label="Selected Point",
                    edgecolor="black",
                    linewidth=1.5,
                )

                plt.draw()
            else:
                print(f"Image not found: {img_path}")

    # Connect the click event
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # df_path = "~/cranfield-navigation-gym/cranavgym/tests/test.pkl"

    # obstacle_map
    # df_path = "~/cranfield-navigation-gym/log_dir/evaluation/0x0_evaluation/PPO_20240916_140744_default_baseline/evaluation_results_raw.pkl"
    # df_path = "~/cranfield-navigation-gym/log_dir/evaluation/camera_repeatability/PPO_20250325_185905_PPO_continuous/evaluation_results_raw.pkl"

    # eval_dir_path = "/home/leo/cranfield-navigation-gym/log_dir/evaluation/camera_noise_update2703_camfail4/"
    # run_name = "PPO_20250327_211107_PPO_no_noise_regime_3x3_eval"

    eval_dir_path = "/home/leo/cranfield-navigation-gym/log_dir/evaluation/camera_noise_update2703_camfail/"
    run_name = "PPO_20250327_191631_PPO_3x3_regime_3x3_eval"

    eval_full_path = os.path.join(eval_dir_path, run_name)

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

    # main(
    #     "~/cranfield-navigation-gym/log_dir/evaluation/camera_repeatability/PPO_20250325_185905_PPO_continuous/evaluation_results_raw.pkl",
    #     None,
    #     "ppo_0x0_eval_0x0_camera_path_Q",
    #     "PPO Trained on 0x0 Noise, Evaluated on 0x0 Noise, Q",
    # )

    main(
        eval_full_path,
        bk_path,
        "ppo_0x0_eval_0x0_camera_path_Q",
        "PPO Trained on 0x0 Noise, Evaluated on 0x0 Noise, Q",
    )

    # main("~/cranfield-navigation-gym/log_dir/evaluation/value_function/PPO_20240925_165730_value_funct_ppo_3x3_default_map_3x3/evaluation_results_raw.pkl", None, "ppo_3x3_eval_3x3_camera_path", "PPO Trained on 3x3 Noise, Evaluated on 3x3 Noise")
