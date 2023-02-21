def reward_function(params):
Example of using steering angle
# Read input variable
steering = abs(params['steering_angle']) # We don't care whether it is left or right steering
# Initialize the reward with typical value reward = 1.0
# Penalize if car steer too much to prevent zigzag STEERING THRESHOLD 20.0 =
if steering > ABS_STEERING_THRESHOLD:
reward* 0.8
return reward









def reward_function(params):
Example of using steps and progress
# Read input variable steps params['steps'] = progress params['progress'] =
# Total num of steps we want the car to finish the lap, it will vary depends on the track Length TOTAL_NUM_STEPS = 300
# Initialize the reward with typical value reward = 1.0
# Give additional reward if the car pass every 100 steps faster than expected if (steps % 100) == 0 and progress> (steps / TOTAL_NUM_STEPS) * 100 : reward + 10.0
return reward