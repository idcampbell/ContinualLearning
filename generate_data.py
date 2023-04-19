mport numpy as np
import matplotlib.pyplot as plt

# Define the world size and resolution
world_size = (2, 2)
resolution = 200

# Define the range of resource sizes and shapes
min_size = 0.5
max_size = 2.0
min_shape = 0.5
max_shape = 2.0

# Define the number of resource gradients to generate
num_resources = 100

fig, axes = plt.subplots(10, 10, sharex=True, sharey=True, tight_layout=True, figsize=(20,20))

# Store all gradients.
all_gradients = np.zeros([num_resources, resolution**2])

# Generate the resource gradients
for i, ax in enumerate(axes.ravel()):
    
    # Randomly generate the resource size, shape, and position
    size = np.random.uniform(min_size, max_size)
    shape = np.random.uniform(min_shape, max_shape)
    pos_x = np.random.uniform(-1, 1)
    pos_y = np.random.uniform(-1, 1)
    
    # Calculate the bounds of the resource gradient based on the resource size and shape
    x_bound = world_size[0]/2 - size*2
    y_bound = world_size[1]/2 - size*2*shape
    
    # Generate the x and y coordinates of the ellipsoid gradient within the bounds and at the specified position
    x = np.linspace(-x_bound, x_bound, resolution) + pos_x
    y = np.linspace(-y_bound, y_bound, resolution) + pos_y
    xx, yy = np.meshgrid(x, y)
    
    # Calculate the distance of each point in the meshgrid from the center of the ellipsoid
    distance = np.sqrt((xx)**2 + ((yy)/shape)**2)
    
    # Calculate the gradient strength as a function of the distance from the center of the ellipsoid
    gradient = np.exp(-(distance**2)/(2*(size**2)))
    all_gradients[i] = gradient.ravel()
    
    # Plot the resource gradient
    ax.imshow(gradient, extent=[-1, 1, -1, 1])


from sklearn.metrics.pairwise import cosine_similarity
test = cosine_similarity(all_gradients)
plt.imshow(test)
