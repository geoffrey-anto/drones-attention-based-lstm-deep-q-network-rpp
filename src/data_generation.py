import numpy as np

def generate_training_data(num_samples, seq_length):
    X = np.zeros((num_samples, seq_length, 6))  # 3 for position, 3 for velocity
    y = np.zeros((num_samples, 3))  # Next position

    for i in range(num_samples):
        # Generate random start and end points
        start = np.random.rand(3) * 100
        end = np.random.rand(3) * 100
        
        # Generate a simple path between start and end
        path = np.linspace(start, end, seq_length)
        
        # Add some noise to make it more realistic
        path += np.random.randn(*path.shape) * 2
        
        # Calculate velocities
        velocities = np.diff(path, axis=0, prepend=path[:1])
        
        # Combine position and velocity
        X[i] = np.hstack((path, velocities))
        
        # Next position is the target
        y[i] = end

    return X, y
