# QRI Project at TRiCAM


	# Find mean and standard deviation of oil production to discover outliers
	oils_array = np.array(oils)
	oils_mean = np.mean(oils_array)
	oils_std = np.std(oils_array)
	
# 	# Remove zeroed data points and outliers
# 	i = 0
# 	while i < len(oils):
# 		OUTLIER_Z = 3.5
# 		if abs(oils[i] - oils_mean) > OUTLIER_Z * oils_std:
# 			del dates[i]
# 			del oils[i]
# 		else:
# 			i += 1
