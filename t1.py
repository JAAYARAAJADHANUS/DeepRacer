import math

def straight_line_reward(current_reward, steering):
    # Positive reward if the car is in a straight line going fast
    if abs(steering) < 0.1:
        current_reward *= 1.5
    elif abs(steering) < 0.2:
        current_reward *= 1.2
    elif steering > 15:
        current_reward *= 0.8
    elif steering > 30:
        current_reward *= 0.5
        
    return current_reward
def direction_reward(reward, waypoints, closest_waypoints, heading):
    DIRECTION_THRESHOLD = 10
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]

    # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
    direction = math.degrees(math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0]))

    # Cacluate difference between track direction and car heading angle
    direction_diff = abs(direction - heading)

    # Penalize if the difference is too large
    malus=1
    
    if direction_diff > DIRECTION_THRESHOLD:
        malus=1-(direction_diff/50)
        if malus<0 or malus>1:
            malus = 0
        reward *= malus
    
    return reward
    

def reward_function(params):
    '''
    Use square root for center line
    '''
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    steering = abs(params['steering_angle'])
    speed = params['speed']
    all_wheels_on_track = params['all_wheels_on_track']
    reward = 0.001
    if params['all_wheels_on_track']:
        reward = 1 - (distance_from_center / (track_width/2))**(4)
        if reward < 0:
            reward = 0.001
        reward = straight_line_reward(reward, steering)
        reward = direction_reward(reward, params['waypoints'], params['closest_waypoints'], params['heading'])
        reward *= (1+params['progress']/100)
            
    else:
        reward = 0.001

    return float(reward)