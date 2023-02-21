define reward_function(params):

	#Example of using all_wheels_on_track and speed
	# Read input variables
	all_wheels_on_track = params['all_wheels_on_track'] 
	speed = params['speed']
	
	# Set the speed threshold based your action space SPEED_THRESHOLD = 0.8
	if not all_wheels_on_track:
	# Penalize if the car goes off track 
	reward = 1e-3 
	
	# Penalize if the car goes too slow
	elif speed < SPEED_THRESHOLD:
		reward = 0.2
	else:
	# High reward if the car stays on track and goes fast 
		reward = 10.0
	return reward