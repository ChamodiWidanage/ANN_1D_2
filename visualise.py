import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# defining inputs
EXPLOSIVE_MASS = 10
DISTANCE_EXPLOSIVE_TO_SURFACE = 10
CORNER_COORDINATES_OF_SURFACE = ([-4, -4], [4, -4], [-4, 4], [4, 4])  # ll, lr, ul, ur
EXPLOSIVE_POSITION = [0, 0]
MESH_SIZE = 0.01


def generate_surface_coordinates(corner_coordinates, mesh_size):
    ll, lr, ul, ur = corner_coordinates

    # Generate grid of coordinates with mesh size
    x_range = np.arange(ll[0], lr[0] + mesh_size, mesh_size)
    y_range = np.arange(ll[1], ul[1] + mesh_size, mesh_size)[::-1]  # Reverse y_range to match the matrix orientation
    x_coords, y_coords = np.meshgrid(x_range, y_range)

    # Create a 2D matrix with coordinates in corresponding positions
    surface_matrix = np.empty_like(x_coords, dtype=object)
    for i in range(surface_matrix.shape[0]):
        for j in range(surface_matrix.shape[1]):
            surface_matrix[i, j] = (x_coords[i, j], y_coords[i, j])

    return surface_matrix

def calculate_angles(surface_matrix, distance_explosive_to_surface):
    origin_x, origin_y = 0, 0

    angle_surface_matrix = np.empty(surface_matrix.shape, dtype=object)
    for i in range(surface_matrix.shape[0]):
        for j in range(surface_matrix.shape[1]):
            distance_from_origin_to_point = np.sqrt((surface_matrix[i, j][0] - origin_x) ** 2 + (surface_matrix[i, j][1] - origin_y) ** 2)
            angle_surface_matrix[i, j] = np.arctan(distance_from_origin_to_point/distance_explosive_to_surface)  # find the angle
    return angle_surface_matrix

def calculate_distances(surface_matrix, distance_explosive_to_surface):
    origin_x, origin_y = 0, 0

    distance_surface_matrix = np.empty(surface_matrix.shape, dtype=object)
    for i in range(surface_matrix.shape[0]):
        for j in range(surface_matrix.shape[1]):
            distance_from_origin_to_point = np.sqrt((surface_matrix[i, j][0] - origin_x) ** 2 + (surface_matrix[i, j][1] - origin_y) ** 2)
            distance_surface_matrix[i, j] = np.sqrt(distance_explosive_to_surface ** 2 + distance_from_origin_to_point ** 2)  # find the distance
    return distance_surface_matrix

def generate_a_random_2d_matrix(width, height):
    random_matrix = np.random.rand(height, width)
    return random_matrix

def generate_heatmap(random_heatmap_df, surface_coordinates_df):
    # Reshape the random heatmap DataFrame to match the surface_coordinates_df
    heatmap_values = random_heatmap_df.values.reshape(surface_coordinates_df.shape[0], surface_coordinates_df.shape[1])

    # Create heatmap with x and y coordinates as axis ticks
    plt.imshow(heatmap_values, cmap='viridis', extent=[surface_coordinates_df[0, 0][0],
                                                       surface_coordinates_df[-1, -1][0],
                                                       surface_coordinates_df[-1, -1][1],
                                                       surface_coordinates_df[0, 0][1]])
    plt.colorbar(label='Value')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Heatmap of Random Values on Surface')
    plt.show()

# Generate surface coordinates
surface_coordinates = generate_surface_coordinates(CORNER_COORDINATES_OF_SURFACE, MESH_SIZE)

# calculate the angles or distances to the pints in the surface from the explosive location
surface_coordinates_angles = calculate_angles(surface_coordinates, DISTANCE_EXPLOSIVE_TO_SURFACE)
surface_coordinates_distances = calculate_distances(surface_coordinates, DISTANCE_EXPLOSIVE_TO_SURFACE)

# Generate random values for heatmap
random_heatmap_values = generate_a_random_2d_matrix(surface_coordinates.shape[1], surface_coordinates.shape[0])

# Convert random matrix to DataFrame
random_heatmap_df = pd.DataFrame(random_heatmap_values.flatten(), columns=['value'])

# Generate heatmap
generate_heatmap(random_heatmap_df, surface_coordinates)

print(surface_coordinates)