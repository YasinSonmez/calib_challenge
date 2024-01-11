import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

# Load video and initialize parameters
focal_length = 910  # Provided focal length

# Function to draw optical flow vectors on the frame
def draw_flow(frame, flow, step=16):
    h, w = frame.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    vis = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)  # Convert to RGB
    cv2.polylines(vis, lines, 0, (0, 255, 0))  # Green color for polylines

    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

    return vis  # Return the modified frame

# Function to draw optical flow vectors with angles and magnitudes
def draw_flow_with_angles(frame, angles, magnitudes, step=16, magnitude_scale=5):
    h, w = frame.shape[:2]

    # Create a grid of points with specified step size
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)

    # Calculate the end points based on angles and magnitude scale
    end_x = x + magnitude_scale * magnitudes[y, x] * np.cos(angles[y, x])
    end_y = y + magnitude_scale * magnitudes[y, x] * np.sin(angles[y, x])

    # Convert to integer for drawing
    x, y, end_x, end_y = map(np.int32, [x, y, end_x, end_y])

    # Ensure that end points are within the image bounds
    mask = (end_x < w) & (end_y < h) & (end_x >= 0) & (end_y >= 0)
    x, y, end_x, end_y = x[mask], y[mask], end_x[mask], end_y[mask]

    lines = np.column_stack((x, y, end_x, end_y)).reshape(-1, 2, 2)

    vis = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)  # Convert to RGB
    cv2.polylines(vis, lines, 0, (0, 255, 0))  # Green color for polylines

    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

    return vis  # Return the modified frame

# Function to preprocess the frame (resize, convert to grayscale)
def preprocess_frame(frame, resize_factor=0.5):
    # Resize the frame
    resized_frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)
    plt.show()
    resized_frame = resized_frame[lower_cut:upper_cut, :]

    # Convert to grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    return gray_frame

# Function to calculate angles and filtered magnitudes from optical flow
def calculate_angles_magnitudes(flow):
    # Calculate magnitudes of flow vectors
    magnitudes = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    flattened_magnitudes = magnitudes.flatten()

    # Sort indices based on magnitudes
    cutoff = np.sort(flattened_magnitudes)[3*len(flattened_magnitudes)//4]

    # Calculate angles from the selected flow vectors
    angles = np.arctan2(flow[..., 1], flow[..., 0])

    # Create a mask to keep only the selected vectors
    maginitude_mask = magnitudes > cutoff

    # Filter if angle is looking towards the mid
    angle_mask = absolute_angle_difference(np.abs(angles_to_mid - angles)) < np.pi/2
    mask = maginitude_mask & angle_mask

    # Apply the filter to magnitudes
    magnitudes_filtered = np.zeros_like(magnitudes)
    magnitudes_filtered[mask] = magnitudes[mask]

    return angles, magnitudes_filtered

# Function to compute the absolute angle difference
def absolute_angle_difference(difference_array):
    return np.minimum(difference_array, 2*np.pi - difference_array)

# Function to precompute angles to optimize convergence point calculation
def precompute_angles(image_width, image_height):
    x_coords = np.arange(2 * image_width)
    y_coords = np.arange(2 * image_height)
    x, y = np.meshgrid(x_coords, y_coords)
    angles_to_origin = np.arctan2(y - image_height, x - image_width)
    return angles_to_origin

# Function to get angles to a specified point from precomputed angles
def get_angles_to_point(angles_to_origin, given_point, image_width, image_height):
    given_x, given_y = given_point
    angles_to_point = angles_to_origin[image_height-given_y:2*image_height-given_y, image_width-given_x:2*image_width-given_x]
    return angles_to_point

# Function to find the convergence point based on flow angles and magnitudes
def find_convergence_point(angles, magnitudes, image_shape, precomputed_angles):
    # Initialize the 2D array to store loss values for each point
    stride = 1
    lower_x = 3*image_shape[1]//8
    upper_x = 5*image_shape[1]//8
    lower_y = 3*image_shape[0]//8
    upper_y = 5*image_shape[0]//8

    loss_array = np.zeros((math.ceil((upper_y-lower_y) / stride), math.ceil((upper_x-lower_x) / stride)))
    # Iterate over each point in the image and compute the total loss
    for x in range(lower_x, upper_x, stride):
        for y in range(lower_y, upper_y, stride):
            convergence_point = (x, y)
            angles_to_point = get_angles_to_point(precomputed_angles, convergence_point, image_shape[1], image_shape[0])
            difference_array = angles_to_point - angles

            mask = (magnitudes != 0)

            total_loss = np.sum(absolute_angle_difference(np.abs(difference_array[mask])))
            # Assign the total_loss to the corresponding location in the loss array
            loss_array[(y - lower_y) // stride, (x - lower_x) // stride] = total_loss

    # Find the convergence point with minimum loss
    best_convergence_point = np.unravel_index(np.argmin(loss_array), loss_array.shape)
    best_convergence_point = (best_convergence_point[1] * stride + lower_x, best_convergence_point[0] * stride + lower_y)

    return np.array(best_convergence_point)

# Function to calculate camera angles based on displacement
def calculate_camera_angles(dx, dy, focal_length):
    yaw = np.arctan2(dx, focal_length)
    pitch = np.arctan2(dy, focal_length)

    return pitch, yaw

if __name__ == "__main__":
    # Main loop for processing frames
    video_number = int(sys.argv[1])
    cap = cv2.VideoCapture('labeled/' + str(video_number) + '.hevc')
    prev_frame = None
    width = 582
    height = 437
    lower_cut = 100
    upper_cut = 310
    modified_height = upper_cut - lower_cut
    precomputed_angles = precompute_angles(width, modified_height)
    mid = (width//2, modified_height//2)
    angles_to_mid = get_angles_to_point(precomputed_angles, mid, width, modified_height)

    i = 0
    result = []
    while i<1200:
        i = i+1
        ret, frame = cap.read()
        frame = preprocess_frame(frame)

        if not ret:
            break

        if prev_frame is not None:
            # Calculate pitch and yaw within the estimate_direction function
            flow = cv2.calcOpticalFlowFarneback(
                prev=prev_frame,
                next=frame,
                flow=None,
                pyr_scale=0.5,
                levels=5,
                winsize=13,
                iterations=10,
                poly_n=5,
                poly_sigma=1.1,
                flags=0
            )

            angles, magnitudes = calculate_angles_magnitudes(flow)
            # Find the convergence point without using minimization
            convergence_point = find_convergence_point(angles, magnitudes, frame.shape[:2], precomputed_angles)
            dx = convergence_point[0] - width/2
            dy = lower_cut + convergence_point[1] - height/2
            pitch, yaw = calculate_camera_angles(dx, -dy, focal_length/2)
            print(i, convergence_point, (dx, dy), (pitch, yaw))
            result.append(np.array([pitch, yaw]))

            # Visualize flow (optional)
            visualized_flow = draw_flow(frame.copy(), flow)
            visualized_flow_with_angles = draw_flow_with_angles(frame.copy(), angles, magnitudes)

            # Draw a marker at the convergence point
            cv2.circle(visualized_flow, convergence_point, 10, (255, 0, 0), -1)

            plt.imshow(visualized_flow)
            plt.show()

            # Draw a marker at the convergence point
            cv2.circle(visualized_flow_with_angles, convergence_point, 10, (255, 0, 0), -1)

            plt.imshow(visualized_flow_with_angles)
            plt.show()
            cv2.waitKey(1)

        # Store the preprocessed grayscale frame for the next iteration
        prev_frame = frame

    data = np.array(result)
    # Save the array to a text file
    np.savetxt('output' + str(video_number) + '.txt', data, fmt='%f', delimiter=' ')