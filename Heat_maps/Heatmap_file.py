import uuid

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import pandas as pd

# defining inputs
EXPLOSIVE_MASS = 8
DISTANCE_EXPLOSIVE_TO_SURFACE = 4
CORNER_COORDINATES_OF_SURFACE = ([-4, -4], [4, -4], [-4, 4], [4, 4])  # ll, lr, ul, ur
EXPLOSIVE_POSITION = [0, 0]
MESH_SIZE = 0.01
IS_TNT = False # set this to True if the input dataset is for TNT, and false for CompB
ANN_ANGLE_MODEL_FILE = 'Peak_Reflected_Pressure_ANN_1_nodrop_noscale.h5'
ANN_DISTANCE_MODEL_FILE = 'Peak_Pressure_ANN_4_final.h5'
USE_DISTANCE_MODEL = False # set this to true to use distance for the model input or to false to use angles for the model input
EXPORT_UNIQUE_ANGLES_OR_DISTANCES = True # set this to true to write the file with unique values of angles or distances
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
            angle_radians = np.arctan(distance_from_origin_to_point/distance_explosive_to_surface)  # find the angle
            angle_degrees = np.round(np.degrees(angle_radians),0)  # Convert radians to degrees, define the required precision
            angle_surface_matrix[i, j] = angle_degrees  # Store angle in degrees
    return angle_surface_matrix

def calculate_distances(surface_matrix, distance_explosive_to_surface):
    origin_x, origin_y = 0, 0

    distance_surface_matrix = np.empty(surface_matrix.shape, dtype=object)
    for i in range(surface_matrix.shape[0]):
        for j in range(surface_matrix.shape[1]):
            distance_from_origin_to_point = np.sqrt((surface_matrix[i, j][0] - origin_x) ** 2 + (surface_matrix[i, j][1] - origin_y) ** 2)
            distance_from_explosive_to_point = np.sqrt(distance_explosive_to_surface ** 2 + distance_from_origin_to_point ** 2)  # find the distance
            distance_surface_matrix[i, j] = np.round(distance_from_explosive_to_point*20)/20 # round to nearest 0.5
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
    plt.colorbar(label='Pressure')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Heatmap of Pressure Values on the Surface')
    plt.show()

def generate_x_dataset(surface_coordinates, distance_explosive_to_surface, explosive_mass, isTNT):

    # calculate the angles or distances to the pints in the surface from the explosive location
    if not USE_DISTANCE_MODEL: # use angles
        surface_coordinates_angles_or_distances = calculate_angles(surface_coordinates, distance_explosive_to_surface)
    else: # use distances
        surface_coordinates_angles_or_distances = calculate_distances(surface_coordinates, distance_explosive_to_surface)

    # flatten the surface coordinates/distances into a 1D array
    surface_coordinates_angles_or_distances_flat = surface_coordinates_angles_or_distances.flatten()

    if EXPORT_UNIQUE_ANGLES_OR_DISTANCES: # wheter or not to write the unique values to a file
        file_name = ('unique_distances' if USE_DISTANCE_MODEL else 'unique_angles') + '.csv'
        uniques = np.unique(surface_coordinates_angles_or_distances_flat)
        # Write the unique values to the CSV file
        np.savetxt(file_name, uniques, delimiter=',', fmt='%s')

    # Create a 2D matrix with the specified columns
    num_points = len(surface_coordinates_angles_or_distances_flat)
    # angle model columns  [explosive mass, standoff distance, angle, compB, TNT]
    # distance model columns  [explosive mass, standoff distance, compB, TNT]
    x_dataset = np.zeros((num_points, (4 if USE_DISTANCE_MODEL else 5)))
    x_dataset[:, 0] = explosive_mass  # Set the first column to explosive_mass
    if USE_DISTANCE_MODEL:
        x_dataset[:, 1] = surface_coordinates_angles_or_distances_flat  # Set the third column to surface_coordinates_angles_flat
        x_dataset[:, 2] = np.where(isTNT, 0, 1)  # Set the fourth column: 1 if not isTNT, 0 otherwise
        x_dataset[:, 3] = np.where(isTNT, 1, 0)  # Set the fifth column: 1 if isTNT, 0 otherwise
    else:
        x_dataset[:, 1] = distance_explosive_to_surface  # Set the second column to distance_explosive_to_surface
        x_dataset[:, 2] = surface_coordinates_angles_or_distances_flat  # Set the third column to surface_coordinates_angles_flat
        x_dataset[:, 3] = np.where(isTNT, 0, 1)  # Set the fourth column: 1 if not isTNT, 0 otherwise
        x_dataset[:, 4] = np.where(isTNT, 1, 0)  # Set the fifth column: 1 if isTNT, 0 otherwise

    return x_dataset

def predict_pressures(ann_model_file, x_dataset):
    model = load_model(ann_model_file)
    predictions = model.predict(x_dataset)

    return predictions

# Generate surface coordinates
surface_coordinates = generate_surface_coordinates(CORNER_COORDINATES_OF_SURFACE, MESH_SIZE)

# generate the x dataset
x_dataset = generate_x_dataset(surface_coordinates, DISTANCE_EXPLOSIVE_TO_SURFACE, EXPLOSIVE_MASS, IS_TNT)

# load the ANN model and predict for the x_dataset
predictions = predict_pressures((ANN_DISTANCE_MODEL_FILE if USE_DISTANCE_MODEL else ANN_ANGLE_MODEL_FILE), x_dataset)

predictions_df = pd.DataFrame(predictions.flatten(), columns=['value'])

# Generate heatmap
generate_heatmap(predictions_df, surface_coordinates)

# # Generate random values for heatmap
# random_heatmap_values = generate_a_random_2d_matrix(surface_coordinates.shape[1], surface_coordinates.shape[0])
#
# # Convert random matrix to DataFrame
# random_heatmap_df = pd.DataFrame(random_heatmap_values.flatten(), columns=['value'])
#
# # Generate heatmap
# generate_heatmap(random_heatmap_df, surface_coordinates)
#
# print(surface_coordinates)