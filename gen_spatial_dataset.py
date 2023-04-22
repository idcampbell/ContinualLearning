from os.path import join
import torch
import numpy as np

def gen_resources(world_size, resolution, size_bounds, shape_bounds,
                  pos_x_bounds, pos_y_bounds, num_resources):
    # Store all resource gradients.
    all_gradients = torch.zeros([resolution**2, num_resources])

    min_size, max_size = size_bounds
    min_shape, max_shape = shape_bounds
    min_pos_x, max_pos_x = pos_x_bounds
    min_pos_y, max_pos_y = pos_y_bounds
    
    # Generate the resource gradients
    for i in range(num_resources):
        # Randomly generate the resource size, shape, and position
        size = np.random.uniform(min_size, max_size)
        shape = np.random.uniform(min_shape, max_shape)
        pos_x = np.random.uniform(min_pos_x, max_pos_x)
        pos_y = np.random.uniform(min_pos_y, max_pos_y)

        # Calculate the bounds of the resource gradient based on the resource size and shape
        x_bound = world_size[0]/2 - size*2
        y_bound = world_size[1]/2 - size*2*shape

        # Generate the x and y coordinates of the ellipsoid gradient within the bounds and at the specified position
        x = np.linspace(-x_bound, x_bound, resolution) + pos_x
        y = np.linspace(-y_bound, y_bound, resolution) + pos_y
        xx, yy = np.meshgrid(x, y)

        # Calculate the distance of each point in the meshgrid from the center of the ellipsoid.
        distance = np.sqrt((xx)**2 + ((yy)/shape)**2)

        # Calculate the gradient strength as a function of the distance from the center of the ellipsoid.
        gradient = np.exp(-(distance**2)/(2*(size**2)))
        all_gradients[:,i] = torch.tensor(gradient.ravel())
    
    return all_gradients


if __name__=='__main__':
    # Define the world size and resolution
    world_size = (2, 2)
    resolution = 100

    # Define the number of resource gradients to generate
    num_resources = 100
    
    # Define the range of resource sizes and shapes
    size_bounds, shape_bounds = (.5,2), (.5,2)
    pos_x_bounds, pos_y_bounds = (-1,1), (-1,1)
    size_bounds2, shape_bounds2 = (.05,.1), (.75,1.5)
    pos_x_bounds2, pos_y_bounds2 = (-.75,.75), (-.75,-.25)

    # Generate the resource gradients
    gradients1 = gen_resources(world_size, resolution, size_bounds, shape_bounds,
                               pos_x_bounds, pos_y_bounds, num_resources)
    gradients2 = gen_resources(world_size, resolution, size_bounds2, shape_bounds2,
                               pos_x_bounds2, pos_y_bounds2, num_resources)
    
    # Save out the dataset.
    basepath = './data/'
    x = torch.tile(torch.arange(resolution), (num_resources,)) / resolution
    y = torch.repeat_interleave(torch.arange(resolution), num_resources) / resolution
    inputs = torch.vstack([x,y]).T
    torch.save(inputs, join(basepath, 'spatial_inputs.pt'))
    torch.save(gradients1, join(basepath, 'spatial_outputs.pt'))
    torch.save(gradients2, join(basepath, 'spatial_outputs_narrow.pt'))