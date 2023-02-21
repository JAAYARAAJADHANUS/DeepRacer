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
        print("over_center")
        distance_from_center_reward = 1e-3
    print("distance_from_center_reward: %.2f" % distance_from_center_reward)
    reward = distance_from_center_reward


    # ステアリングを報酬に反映させる
    # 左が正、右が負
    steering_reward = 1e-3
    if distance_from_center > 0:
        if is_left_of_center and steering_angle <= 0:
            steering_reward = 1.0
        elif (not is_left_of_center) and steering_angle >= 0:
            steering_reward = 1.0
    else:
        if steering_angle == 0:
            steering_reward = 1.0
    print("steering_reward: %.2f" % steering_reward)
    reward += steering_reward


    # ジグザク抑制
    # 参考) https://dev.classmethod.jp/machine-learning/aws-deepracer-pattern-of-reward-function/
    # ハンドルの操作角(-30°〜30°)をparamsから取得
    # 操作角の絶対値を計算(右旋回、左旋回問わず角度の大きさで判断する
    steering2_reward = 1.0

    # 急ハンドルを判定する為の閾値を定義して、それ以上の操作角だった場合にペナルティを与える
    # 閾値は行動パターンの設定によって変動する
    STEERING_THRESHOLD = 20.0
    if steering_angle > STEERING_THRESHOLD:
        steering2_reward = 0.8

    print("steering2_reward: %.2f" % steering2_reward)
    reward *= steering2_reward

    print("total reward: %.2f" % reward)
    return float(reward)