def reward_function(params):
    # パラメータ取得
    all_wheels_on_track = params['all_wheels_on_track']
    x = params['x']
    y = params['y']
    distance_from_center = params['distance_from_center']
    is_left_of_center = params['is_left_of_center']
    heading = params['heading']
    progress = params['progress']
    steps = params['steps']
    speed = params['speed']
    steering_angle = params['steering_angle']
    track_width = params['track_width']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']

    reward = 0

    # 車両がトラックラインの外側に出たらペナルティ
    if not all_wheels_on_track:
        print("off_track")
        return 1e-3


    # 車両がトラックの中心に近いほど多くの報酬を返す
    distance_from_center_reward = 0
    marker_1 = 0.1 * track_width
    marker_2 = 0.2 * track_width
    marker_3 = 0.3 * track_width
    if distance_from_center >= 0.0 and distance_from_center <= marker_1:
        distance_from_center_reward = 1
    elif distance_from_center <= marker_2:
        distance_from_center_reward = 0.8
    elif distance_from_center <= marker_3:
        distance_from_center_reward = 0.6
    else:
        return 1e-3

    print("distance_from_center_reward: %.2f" % distance_from_center_reward)
    reward = distance_from_center_reward

    print("total reward: %.2f" % reward)
    return float(reward)