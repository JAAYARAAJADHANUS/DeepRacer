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
    if not on_track:
        print("off_track")
        return 1e-3


    # ステアリングを報酬に反映させる
    steering_reward = 1e-3
    if distance_from_center > 0:
        # (´・ω・`) 逆になってた
        if is_left_of_center and steering >= 0:
            steering_reward = 1.0
        elif (not is_left_of_center) and steering <= 0:
            steering_reward = 1.0
    else:
        if steering == 0:
            steering_reward = 1.0
    print("steering_reward: %.2f" % steering_reward)
    reward += steering_reward

    print("total reward: %.2f" % reward)
    return float(reward)