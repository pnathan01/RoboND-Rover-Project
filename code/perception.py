import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_range_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    limit = 20
    rgb_red_low = rgb_thresh[0] - limit
    rgb_red_hi = rgb_thresh[0] + limit
    rgb_blu_low = rgb_thresh[1] - limit
    rgb_blu_hi = rgb_thresh[1] + limit
    rgb_grn_low = rgb_thresh[2] - limit
    rgb_grn_hi = rgb_thresh[2] + limit
    above_thresh = (img[:,:,0] > rgb_red_low) \
                & (img[:,:,0] < rgb_red_hi) \
                & (img[:,:,1] > rgb_blu_low) \
                & (img[:,:,1] < rgb_blu_hi) \
                & (img[:,:,2] > rgb_grn_low) \
                & (img[:,:,2] < rgb_grn_hi)
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    source = np.float32([[125,95], [205,95], [310,140], [10,140]])
    destination = np.float32([[160,140], [170,140], [170,145], [160,145]])
    # 2) Apply perspective transform
    warped = perspect_transform(Rover.img, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    terrain_red_threshold = 160
    terrain_green_threshold = 160
    terrain_blue_threshold = 160
    terrain_rgb_threshold = (terrain_red_threshold, terrain_green_threshold, terrain_blue_threshold)
    terrain_select = color_thresh(warped, rgb_thresh=terrain_rgb_threshold)
    sample_red_threshold = 125
    sample_green_threshold = 125
    sample_blue_threshold = 6
    sample_rgb_threshold = (sample_red_threshold, sample_green_threshold, sample_blue_threshold)
    sample_select = color_range_thresh(warped, rgb_thresh=sample_rgb_threshold)
    obs_red_threshold = 160
    obs_green_threshold = 160
    obs_blue_threshold = 160
    obs_rgb_threshold = (obs_red_threshold, obs_green_threshold, obs_blue_threshold)
    obs_select = color_thresh(warped, rgb_thresh=obs_rgb_threshold)
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:,:,0] = obs_select*255
    Rover.vision_image[:,:,1] = sample_select*255
    Rover.vision_image[:,:,2] = terrain_select*255
    # 5) Convert map image pixel values to rover-centric coords
    x_pixel, y_pixel = rover_coords(terrain_select)
    x_spixel, y_spixel = rover_coords(sample_select)
    x_opixel, y_opixel = rover_coords(obs_select)
    # 6) Convert rover-centric pixel values to world coordinates
    scale = 10
    x_pix_world, y_pix_world = pix_to_world(x_pixel, y_pixel, Rover.pos[0], Rover.pos[1], Rover.yaw, Rover.worldmap.shape[0], scale)
    x_spix_world, y_spix_world = pix_to_world(x_spixel, y_spixel, Rover.pos[0], Rover.pos[1], Rover.yaw, Rover.worldmap.shape[0], scale)
    x_opix_world, y_opix_world = pix_to_world(x_opixel, y_opixel, Rover.pos[0], Rover.pos[1], Rover.yaw, Rover.worldmap.shape[0], scale)
    # 7) Update Rover worldmap (to be displayed on right side of screen)
    Rover.worldmap[y_opix_world, x_opix_world, 0] += 1
    Rover.worldmap[y_pix_world, x_pix_world, 2] += 1
    Rover.worldmap[y_spix_world, x_spix_world, 1] += 1
    # 8) Convert rover-centric pixel positions to polar coordinates
    rover_centric_pixel_distances, rover_centric_angles = to_polar_coords(x_pixel, y_pixel)
    sample_distances, sample_angles = to_polar_coords(x_spixel, y_spixel)
    obs_distances, obs_angles = to_polar_coords(x_opixel, y_opixel)
    # Update Rover pixel distances and angles
    Rover.nav_dists = rover_centric_pixel_distances
    Rover.nav_angles = rover_centric_angles
    Rover.sample_dists = sample_distances
    Rover.sample_angles = sample_angles
    Rover.obs_dists = obs_distances
    Rover.obs_angles = obs_angles
    return Rover
