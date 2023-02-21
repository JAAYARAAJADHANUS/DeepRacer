def steering_reward(reward, steering, is_left_of_center, speed):
    # For hard turn:
    if (steering < -25 and is_left_of_center == False) or (steering > 25 and is_left_of_center == True):
        reward *= 1.2
    else:
        reward *= 0.8

    # For soft turn
    if abs(steering) < 15 and speed < 4:
        reward *= 1.1
    else:
        reward *= 0.9

    # For straight
    if abs(steering) < 1 and speed > 4:
        reward *= 1.4

    return reward
    
def direction_reward(reward, waypoints, closest_waypoints, heading):
    import math
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]

    # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
    direction = math.degrees(math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0]))

    # Cacluate difference between track direction and car heading angle
    direction_diff = abs(direction - heading)

    # Penalize if the difference is too large
    if direction_diff > 30:
        reward *= 0.5
    else:
        reward *= 1.1

    return reward

    
def straight_line_reward(current_reward, steering, speed):
    # Positive reward if the car is in a straight line going fast
    if abs(steering) < 0.1:
        current_reward *= 2
    elif abs(steering) < 0.3:
        current_reward *= 1.5
    elif abs(steering) < 0.5:
        current_reward *= 1.2
    return current_reward

def reward_function(params):
    '''
    Example of rewarding the agent to follow center line
    '''
    # Give a very low reward by default
    reward = 1e-3
    # Read input parameters
    
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    # Read input parameters
    all_wheels_on_track = params['all_wheels_on_track']
    
    steer = abs(params['steering_angle'])
    steer_thres = 12
    if all_wheels_on_track and (0.5*track_width - distance_from_center) >= 0.05:
        if steer > steer_thres:
            reward = 0.8
        else:
            
            reward = 1.0
        reward = direction_reward(reward,params['waypoints'],params['closest_waypoints'],params['heading'])
        reward = steering_reward(reward,params['steering_angle'],params['is_left_of_center'] , params['speed'])
        reward = straight_line_reward(reward, params['steering_angle'], params['speed'])
    elif all_wheels_on_track and distance_from_center< track_width/4: 
        reward = steering_reward(reward,params['steering_angle'],params['is_left_of_center'] , params['speed'])
    else:
        reward = 1e-3
    
    return float(reward)