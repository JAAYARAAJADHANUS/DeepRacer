#https://everdark.github.io/k9/projects/deepracer_2020/deepracer_2020.html


import plotly.io as pio
pio.renderers.default = "notebook_connected"
# Some paths.
MODEL_DIR = "/home/kylechung/deepracer-local/data/minio/bucket/"
TRACK_URL = "https://github.com/aws-samples/aws-deepracer-workshops/raw/master/log-analysis/tracks/reinvent_base.npy"
def reward_function(params):

    track_width = params["track_width"]
    distance_from_center = params["distance_from_center"]

    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    if distance_from_center <= marker_1:
        reward = 1.0
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3

    return float(reward)
# Define utility function for plotting metrics.
import os
import json
import pandas as pd
import plotly.express as px


def parse_metrics(infile, starting_episode=0):
  with open(infile) as f:
    metrics = pd.DataFrame(json.load(f)["metrics"])
    metrics_t = metrics.query("phase == 'training'").copy()
    metrics_t["episode"] += starting_episode
    metrics_e = metrics.query("phase == 'evaluation'").copy()
    metrics_e["episode"] += starting_episode
  return metrics_t, metrics_e


def plot_metrics(train_metrics, eval_metrics, title=None):
  train_metrics.reset_index(drop=True, inplace=True)
  train_metrics["iter"] = (train_metrics.index / 20 + 1).astype(int)
  eval_metrics["iter"] = (eval_metrics["episode"] / 20).astype(int)
  train_progress = train_metrics.groupby("iter")["completion_percentage"].mean()
  train_progress = train_progress.to_frame().reset_index()
  train_progress["phase"] = "training"
  eval_progress = eval_metrics.groupby("iter")["completion_percentage"].mean()
  eval_progress = eval_progress.to_frame().reset_index()
  eval_progress["phase"] = "evaluation"
  progress = pd.concat([train_progress, eval_progress])
  fig = px.line(progress, x="iter", y="completion_percentage", color="phase",
                labels={"iter": "Training Iteration",
                        "completion_percentage": "Mean Percentage of Completion"},
                title=title)
  return fig
train_metrics_default, eval_metrics_default = parse_metrics(
  os.path.join(MODEL_DIR, "default/TrainingMetrics.json"))
plot_metrics(train_metrics_default, eval_metrics_default)
# Calculate the average elapsed time for lap completion in evaluation.
# Note that for each iteration (20 episodes) there are 6 evaluation run.
# Here we took only the evaluation after training for at least 500 episodes.
eval_metrics_default.query(
  "episode_status == 'Lap complete' and episode >= 500")[
  "elapsed_time_in_milliseconds"].mean() / 1000
import io
import requests
from numpy import load as load_npy


def maybe_download_waypoints(url):
  file = os.path.basename(url)
  if os.path.exists(file):
    wp = load_npy(file)
  else:
    response = requests.get(url)
    wp = load_npy(io.BytesIO(response.content))
  waypoints = wp[:,:2].tolist()
  return waypoints
waypoints = maybe_download_waypoints(TRACK_URL)
import plotly.graph_objects as go


def plot_waypoints(waypoints, annotate=True, title=None):
  if annotate:
    text = [str(i) for i in range(len(waypoints))]
  else:
    text = None
  x, y = zip(*waypoints)
  fig = go.Figure(data=go.Scatter(x=x, y=y, mode="markers+text",
                                  text=text, textposition="bottom center"))
  fig.update_layout(
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False, scaleanchor = "x", scaleratio=1),
    title="AWS DeepRacer re:invent 2018 Track" if title is None else title
  )
  fig.show()
plot_waypoints(waypoints)
def up_sample(waypoints, k):
  p = waypoints
  n = len(p)
  return [[i / k * p[(j+1) % n][0] + (1 - i / k) * p[j][0],
          i / k * p[(j+1) % n][1] + (1 - i / k) * p[j][1]] for j in range(n) for i in range(k)]
# Plot the same track but with 10X denser.
plot_waypoints(up_sample(waypoints, 10), annotate=False,
               title="re:invent 2018 Track Waypoints Up Sampled")
def plot_xy_base(points):
  x, y = zip(*points)
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=[-6, 6], y=[0, 0], mode="lines",
                           line=dict(color="RoyalBlue"), showlegend=False))
  fig.add_trace(go.Scatter(x=[0, 0], y=[-6, 6], mode="lines",
                           line=dict(color="RoyalBlue"), showlegend=False))
  fig.add_trace(go.Scatter(
    x=x, y=y, mode="markers+text", marker=dict(color="black", size=10),
    text=["Current Position (0, 0)", "Next Waypoint (3, 4)"],
    textfont_size=14, textposition="bottom center", showlegend=False)) 
  fig.update_layout(height=600, width=600, title="Car in a Step to Make the Next Move")
  fig.update_xaxes(range=[-5, 5])
  fig.update_yaxes(range=[-5, 5])
  
  # Add direction with arrow.
  fig.add_annotation(dict(
    showarrow=True,
    x=point_a[0], y=point_a[1], ax=point_0[0], ay=point_0[1],
    xref="x", yref="y", axref="x", ayref="y",
    arrowhead=4, arrowsize=2, arrowcolor="red", arrowwidth=2
  ))
  
  # Add the theta symbol.
  fig.add_trace(go.Scatter(
    x=(.75,), y=(.5,), mode="text", text=r"$\theta$",
    textfont=dict(size=20, color="red"), showlegend=False
  ))
  
  return fig
point_0 = (0, 0)  # Assume this is our current position.
point_a = (3, 4)  # Assume this is the closet next waypoint on the track.

plot_xy_base([point_0, point_a])
import math


def angle(x, y):
  a = math.degrees(math.atan2(
    y[1] - x[1],
    y[0] - x[0]
  ))
  return a
angle(point_0, point_a)  # Solve for theta in the plot.
def heading_point(p, heading, r):
  h = (
    p[0] + r * math.cos(math.radians(heading)),
    p[1] + r * math.sin(math.radians(heading))
  )
  return h
hp = heading_point(point_0, 110, 5)
hp
fig = plot_xy_base([point_0, point_a])
fig.add_trace(go.Scatter(
  x=(hp[0],), y=(hp[1],), mode="markers+text", marker=dict(color="black", size=10),
  text=["Heading Point (?, ?)"],
  textfont_size=14, textposition="bottom center", showlegend=False))
fig.add_annotation(dict(
  showarrow=True,
  x=hp[0], y=hp[1], ax=point_0[0], ay=point_0[1],
  xref="x", yref="y", axref="x", ayref="y",
  arrowhead=4, arrowsize=2, arrowcolor="orange", arrowwidth=2
))
fig.show()
angle(point_0, point_a) - angle(point_0, (-3, -4))  # Assuming heading for the opposite direction.
def score_heading_delta(current_point, heading_point, desired_point):
  desired = angle(current_point, desired_point)
  heading = angle(current_point, heading_point)
  return 1 - abs((desired - heading) / 180)


# Possible rewards given the specific state illustrated in the above plot.
some_headings = list(range(-180, 180, 10))
heading_points = [heading_point(point_0, h, 5) for h in some_headings]
possible_rewards_1 = [score_heading_delta(point_0, h, point_a) for h in heading_points]
import plotly.express as px  # Let's use the higher-level API this time.


px.scatter(x=some_headings, y=possible_rewards_1,
           labels={"x": "Possible Heading Delta (in Degrees)", "y": "Reward"})
def dist(p1, p2):
  return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def score_heading_vector_delta(current_point, heading_point, desired_point):
  heading_r = dist(current_point, heading_point)
  desired_r = dist(current_point, desired_point)
  delta_r = dist(heading_point, desired_point)
  return 1 - (delta_r / (desired_r * 2))


possible_rewards_2 = [score_heading_vector_delta(point_0, h, point_a) for h in heading_points]
px.scatter(x=some_headings, y=possible_rewards_2,
           labels={"x": "Possible Heading Delta (in Degrees)", "y": "Reward"})
def reward_function(params):

    x, y = params["x"], params["y"]
    all_wheels_on_track = params["all_wheels_on_track"]
    waypoints = params["waypoints"]
    heading = params["heading"]
    next_waypoint = waypoints[params["closest_waypoints"][1]]

    reward = 1e-3

    if all_wheels_on_track:
        r = math.hypot(x - next_waypoint[0], y - next_waypoint[1])
        heading_point = heading_point((x, y), heading, r)
        delta = math.hypot(heading_point[0] - next_waypoint[0],
                           heading_point[1] - next_waypoint[1])
        reward += (1 - (delta / (r * 2)))

    return reward
import math


def dist(x, y):
    return math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)


def angle(x, y):
    a = math.degrees(math.atan2(
        y[1] - x[1],
        y[0] - x[0]
    ))
    return a


def up_sample(waypoints, k):
    p = waypoints
    n = len(p)
    return [[i / k * p[(j+1) % n][0] + (1 - i / k) * p[j][0],
             i / k * p[(j+1) % n][1] + (1 - i / k) * p[j][1]] for j in range(n) for i in range(k)]


def closest_waypoint_ind(p, waypoints):
    distances = [dist(wp, p) for wp in waypoints]
    min_dist = min(distances)
    return distances.index(min_dist)


def score_delta_steering(delta, worst=60):
    return max(1 - abs(delta / worst), 0)


def reward_function(params):

    reward = 1e-3

    # Read enviroment paramters.
    waypoints = params["waypoints"]
    track_width = params["track_width"]

    # Read states
    x, y = params["x"], params["y"]
    heading = params["heading"]
    steering_angle = params["steering_angle"]

    # Up-sample waypoints to form a series of dense racing line points.
    waypoints = up_sample(waypoints, k=30)

    # Get the closest waypoint given current position (x, y).
    which_closest = closest_waypoint_ind((x, y), waypoints)

    # Re-order the waypoints from the cloest for latter lookup.
    following_waypoints = waypoints[which_closest:] + waypoints[:which_closest]

    # Determine the desired heading angle based on a target waypoint.
    # 1. Locate the target waypoint with a search radius.
    #    Target point should be the cloest waypoint just outside the radious.
    search_radius = track_width * 0.9
    target_waypoint = waypoints[which_closest]
    for i, p in enumerate(following_waypoints):
        if dist(p, (x, y)) > search_radius:
            target_waypoint = following_waypoints[i]
            break
    # 2. Determine the desired steering angle.
    target_heading = angle((x, y), target_waypoint)
    target_steering = target_heading - heading
    delta_steering = steering_angle - target_steering

    # Reward based on difference between current and desired steering_angle.
    reward += score_delta_steering(delta_steering, worst=45)

    return float(reward)
ssible_steering_deltas = list(range(-180, 180, 5))
possible_rewards_3 = [score_delta_steering(d, worst=45) for d in possible_steering_deltas]
px.scatter(x=possible_steering_deltas, y=possible_rewards_3,
           labels={"x": "Possible Steering Delta (in Degrees)", "y": "Reward"})
def reward_function(params):
    reward = 1e-3
    reward += params["speed"]
    return reward
def is_near_straight(waypoints, k=120):
    angles = []
    for i in range(k):
        angles.append(angle(waypoints[i], waypoints[i + 1]))
    mean = sum(angles) / len(angles)
    sd = math.sqrt(sum([(x - mean)**2 for x in angles]) / len(angles))
    return sd <= 0.01
[
{
    "steering_angle": -30,
    "speed": 1,
    "index": 0
},
{
    "steering_angle": -30,
    "speed": 2,
    "index": 1
},
{
    "steering_angle": -30,
    "speed": 3,
    "index": 2
},
{
    "steering_angle": -15,
    "speed": 1,
    "index": 3
},
{
    "steering_angle": -15,
    "speed": 2,
    "index": 4
},
{
    "steering_angle": -15,
    "speed": 3,
    "index": 5
},
{
    "steering_angle": 0,
    "speed": 1,
    "index": 6
},
{
    "steering_angle": 0,
    "speed": 2,
    "index": 7
},
{
    "steering_angle": 0,
    "speed": 3,
    "index": 8
},
{
    "steering_angle": 15,
    "speed": 1,
    "index": 9
},
{
    "steering_angle": 15,
    "speed": 2,
    "index": 10
},
{
    "steering_angle": 15,
    "speed": 3,
    "index": 11
},
{
    "steering_angle": 30,
    "speed": 1,
    "index": 12
},
{
    "steering_angle": 30,
    "speed": 2,
    "index": 13
},
{
    "steering_angle": 30,
    "speed": 3,
    "index": 14
}
]
%%bash
# The local robomaker container doesn't seem to output simulation log for evaluation phase.
# So this log is downloaded from an evaluation run on DeepRacer Console.
cat ~/Downloads/robo.log | grep SIM_TRACE_LOG > /tmp/robo.log
import pandas as pd

# Be aware that the reward number is calculated by a default function when the simulation is for evaluation run.
sim_logs = pd.read_csv("/tmp/robo.log", header=None)
sim_logs.columns = [
  "episode",
  "step",
  "x",
  "y",
  "heading",
  "steering_angle",
  "speed",
  "action_taken",
  "reward",
  "job_completed",
  "all_wheels_on_track",
  "progress",
  "closest_waypoint_index",
  "track_length",
  "time",
  "status"
]
sim_logs.head()
def action_count(df):
  cnt = df.groupby("action_taken").size().to_frame(name="frequency").reset_index()
  cnt["pct"] = cnt["frequency"] / cnt["frequency"].sum()
  return cnt

act_cnt = action_count(sim_logs[sim_logs["episode"].str.endswith("0")])
fig = px.bar(act_cnt, x="action_taken", y="frequency", text="pct",
            title="Action Distribution for a Successful Lap")
fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
fig
train_metrics_0, eval_metrics_0 = parse_metrics(
  os.path.join(MODEL_DIR, "a21-base/TrainingMetrics.json"))
plot_metrics(train_metrics_0, eval_metrics_0,
            title="Training Progress on Model with 21 Actions: First 60 Iterations")
train_metrics_1, eval_metrics_1 = parse_metrics(
  os.path.join(MODEL_DIR, "a21-120/TrainingMetrics.json"),
  starting_episode=1200)
train_metrics_2, eval_metrics_2 = parse_metrics(
  os.path.join(MODEL_DIR, "a21-180/TrainingMetrics.json"),
  starting_episode=2400)
train_metrics = pd.concat([train_metrics_0, train_metrics_1, train_metrics_2])
eval_metrics = pd.concat([eval_metrics_0, eval_metrics_1, eval_metrics_2])
plot_metrics(train_metrics, eval_metrics,
            title="Training Progress on Model with 21 Actions")
plot_metrics(*parse_metrics(
  os.path.join(MODEL_DIR, "DeepRacer-Metrics/TrainingMetrics-default-05.json")),
            title="A Model Failed to Learn Fast with a Low Discount Factor")