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

    import math

    reward = 0

    # 車両がトラックラインの外側に出たらペナルティ
    if not all_wheels_on_track:
        print("off_track")
        return 1e-3

    # waypointに向かってすすめ
    # 参考) https://dev.classmethod.jp/machine-learning/aws-deepracer-pattern-of-reward-function/
    # 規定の報酬を設定する
    waypoints_reward = 1.0

    # 現在の位置から最も近い次のwaypointと前のwaypointを取得する
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]

    # 前のwaypointから次のwaypointに向かう角度(radian)を計算する
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0]) 
    # degreeに変換
    track_direction = math.degrees(track_direction)
    print("track_direction: %.2f" % track_direction)

    # コース上の基準軸に対する車体の向きと直近のwaypointを繋ぐ向きの差分を取る
    direction_diff = abs(track_direction - heading)
    print("direction_diff: %.2f" % direction_diff)

    # 算出した方向の差分から車体の向きが大きくズレている場合にペナルティを与える
    # 閾値の設定はコースの種類によって調整が必要
    DIRECTION_THRESHOLD = 10.0
    if direction_diff > DIRECTION_THRESHOLD:
        waypoints_reward = 0.5

    print("waypoints_reward: %.2f" % waypoints_reward)
    reward = waypoints_reward

    return float(reward)