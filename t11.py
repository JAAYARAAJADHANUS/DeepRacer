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

    # ジグザク抑制
    # 参考) https://dev.classmethod.jp/machine-learning/aws-deepracer-pattern-of-reward-function/
    # ハンドルの操作角(-30°〜30°)をparamsから取得
    # 操作角の絶対値を計算(右旋回、左旋回問わず角度の大きさで判断する
    steering2_reward = 1.0

    # 急ハンドルを判定する為の閾値を定義して、それ以上の操作角だった場合にペナルティを与える
    # 閾値は行動パターンの設定によって変動する
    STEERING_THRESHOLD = 20.0
    if steering_angle > STEERING_THRESHOLD:
        steering2_reward = 1e-3

    print("steering2_reward: %.2f" % steering2_reward)
    reward += steering2_reward

    print("total reward: %.2f" % reward)
    return float(reward)