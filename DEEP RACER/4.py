def reward_function(params):
import math
reward = 1e-3
rabbit = [0,0]
pointing = [0,0]
# Read input variables
waypoints params['waypoints'] closest_waypoints = params['closest_waypoints'] heading = params['heading']
x=params['x']
y=params['y']
# Reward when yaw (car_orientation) is pointed to the next waypoint IN FRONT.
# Find nearest waypoint coordinates
rabbit = waypoints [closest_waypoints[1]]
radius = math.hypot(x - rabbit[0], y - rabbit[1])
pointing[0] = x + (radius * math.cos (heading))
pointing[1] = y + (radius math.sin(heading))
vector_delta = math.hypot (pointing[0] - rabbit[0], pointing[1] - rabbit[1])
# Max distance for pointing away will be the radius * 2
# Min distance means we are pointing directly at the next waypoint
# We can setup a reward that is a ratio to this max.
if vector_delta == 0:
reward += 1
else:
reward += ( 1 ( vector_delta / (radius - *
return reward