# use this https://docs.python.org/3/library/configparser.html instead
import configparser

config = configparser.ConfigParser()

config["ROS"] = {"launchfile": "", "port": "11311", "step_pause_time_delta": "0.001"}

config["Map"] = {
    "min_x": "-5",
    "max_x": "5",
    "min_y": "-5",
    "max_y": "5",
}

config["Camera"] = {
    "img_width": "160",
    "img_height": "160",
}

config["Lidar"] = {"lidar_dim": "20"}


if __name__ == "__main__":
    print(f"{config}")
