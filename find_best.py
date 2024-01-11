################################################################################
# Since the predictions from optical flow is too noisy this will test different 
# filtering methods, and compare score of different methods
################################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import pywt
from scipy.signal import savgol_filter
from pykalman import KalmanFilter
from sklearn.model_selection import ParameterGrid

#Determine the overall error score on all videos
def get_mse(gt, test):
    test = np.nan_to_num(test)
    return np.mean(np.nanmean((gt - test)**2, axis=0))

def score(predictions):
    zero_mses = []
    mses = []

    for i in range(0,5):
        gt = np.loadtxt('./labeled/' + str(i) + '.txt')
        mask = ~np.isnan(gt).any(axis=1)
        gt = gt[mask]
        zero_mses.append(get_mse(gt, np.zeros_like(gt)))

        test = predictions[i][mask]
        mses.append(get_mse(gt, test))

    percent_err_vs_all_zeros = 100*np.mean(mses)/np.mean(zero_mses)
    print(f'YOUR ERROR SCORE IS {percent_err_vs_all_zeros:.2f}% (lower is better)')
    return percent_err_vs_all_zeros

def score(predictions):
    zero_mses_dim1 = []
    zero_mses_dim2 = []
    mses_dim1 = []
    mses_dim2 = []

    for i in range(0, 5):
        gt = np.loadtxt('./labeled/' + str(i) + '.txt')
        mask = ~np.isnan(gt).any(axis=1)
        gt = gt[mask]
        zero_mses_dim1.append(get_mse(gt[:, 0], np.zeros_like(gt[:, 0])))
        zero_mses_dim2.append(get_mse(gt[:, 1], np.zeros_like(gt[:, 1])))

        test_dim1 = predictions[i, :, 0][mask]
        test_dim2 = predictions[i, :, 1][mask]

        mses_dim1.append(get_mse(gt[:, 0], test_dim1))
        mses_dim2.append(get_mse(gt[:, 1], test_dim2))

    # Calculate percentage error for each dimension separately
    percent_err_dim1 = 100 * np.mean(mses_dim1) / np.mean(zero_mses_dim1)
    percent_err_dim2 = 100 * np.mean(mses_dim2) / np.mean(zero_mses_dim2)

    # Calculate combined percentage error
    percent_err_combined = 100 * np.mean(mses_dim1 + mses_dim2) / np.mean(zero_mses_dim1 + zero_mses_dim2)

    print(f'YOUR ERROR SCORE (DIMENSION 1) IS {percent_err_dim1:.2f}% (lower is better)')
    print(f'YOUR ERROR SCORE (DIMENSION 2) IS {percent_err_dim2:.2f}% (lower is better)')
    print(f'YOUR COMBINED ERROR SCORE IS {percent_err_combined:.2f}% (lower is better)')

    return percent_err_combined, percent_err_dim1, percent_err_dim2


def interpolate_to_original_size(original_size, filtered_data, axis):
    # Interpolate the filtered data to match the original size
    interp_function = interp1d(np.arange(filtered_data.shape[axis]), filtered_data, kind='linear', axis=axis, fill_value='extrapolate')
    return interp_function(np.linspace(0, filtered_data.shape[axis] - 1, original_size))

def filter_time_series(data, method='moving_average', axis=1, **kwargs):
    """
    Filter time series data using different methods.

    Parameters:
    - data: NumPy array, input time series data (2D).
    - method: str, filtering method ('moving_average', 'ema', 'lowpass', 'median', 'wavelet').
    - axis: int, axis along which to perform the filtering.
    - kwargs: Additional parameters for each filtering method.

    Returns:
    - NumPy array, filtered time series data.
    """
    if method == 'base':
        return data

    elif method == 'moving_average':
        window_size = kwargs.get('window_size', 100)
        return np.apply_along_axis(lambda x: np.convolve(x, np.ones(window_size) / window_size, mode='valid'), axis, data)

    elif method == 'ema':
        alpha = kwargs.get('alpha', 0.01)
        def ema_filter(x):
            result = np.zeros_like(x, dtype=float)
            result[0] = x[0]
            for i in range(1, len(x)):
                result[i] = alpha * x[i] + (1 - alpha) * result[i - 1]
            return result
        return np.apply_along_axis(ema_filter, axis, data)

    elif method == 'lowpass':
        cutoff_frequency = kwargs.get('cutoff_frequency', 0.03)
        sampling_rate = kwargs.get('sampling_rate', 2.0)
        order = kwargs.get('order', 3)
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_frequency / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return np.apply_along_axis(lambda x: filtfilt(b, a, x), axis, data)

    elif method == 'median':
        window_size = kwargs.get('window_size', 10)
        return np.apply_along_axis(lambda x: np.median(np.lib.stride_tricks.sliding_window_view(x, (window_size,)), axis=1), axis, data)
    
    elif method == 'median400_850':
        a = np.apply_along_axis(lambda x: np.median(np.lib.stride_tricks.sliding_window_view(x, (400,)), axis=1), axis, data)
        a = interpolate_to_original_size(1200, a, axis=1)
        b = np.apply_along_axis(lambda x: np.median(np.lib.stride_tricks.sliding_window_view(x, (850,)), axis=1), axis, data)
        b = interpolate_to_original_size(1200, b, axis=1)
        result = np.concatenate((b[:, :, 0, np.newaxis], a[:, :, 1, np.newaxis]), axis=2)
        for i, matrix in enumerate(result):
            # Generate a filename based on the index (i)
            j = i+5
            filename = f'submit/{j}.txt'
            
            # Save the matrix to a text file
            np.savetxt(filename, matrix, fmt='%f', delimiter=' ')
        return result

    elif method == 'wavelet':
        wavelet = kwargs.get('wavelet', 'db4')
        level = kwargs.get('level', 6)
        threshold = kwargs.get('threshold', 0.3)
        def wavelet_filter(x):
            coeffs = pywt.wavedec(x, wavelet, level=level)
            coeffs[1:] = (pywt.threshold(c, value=threshold, mode='soft') for c in coeffs[1:])
            return pywt.waverec(coeffs, wavelet)
        return np.apply_along_axis(wavelet_filter, axis, data)

    elif method == 'kalman':
        process_variance = kwargs.get('process_variance', 1e-4)
        measurement_variance = kwargs.get('measurement_variance', 1)
        n_samples, n_frames, n_features = data.shape

        filtered_state_means = np.zeros_like(data)

        for i in range(n_samples):
            # Ensure initial_state_mean has the correct shape
            initial_state_mean = np.full(n_features, fill_value=data[i].mean())

            # Create Kalman filter model
            kf = KalmanFilter(
                initial_state_mean=initial_state_mean,
                initial_state_covariance=np.eye(n_features),
                observation_covariance=np.eye(n_features) * measurement_variance,
                transition_covariance=np.eye(n_features) * process_variance
            )

            # Apply the filter to each vector independently
            filtered_state_means[i, :, :], _ = kf.filter(data[i, :, :])

        return filtered_state_means

    elif method == 'savgol':
        window_size = kwargs.get('window_size', 30)
        order = kwargs.get('order', 5)
        return np.apply_along_axis(lambda x: savgol_filter(x, window_size, order), axis, data)
    
    elif method == 'average_methods':
        avg_size = kwargs.get('avg_size', 10)
        ranked_results = sorted(results.items(), key=lambda x: x[1][0])
        top_n_methods = [result[1][1] for result in ranked_results[:avg_size]]
        average_predictions = np.mean(top_n_methods, axis=0)
        return average_predictions
    
    elif method == 'constant_average':
        average = np.mean(data, axis=1, keepdims=True)
        print(average.shape)
        constant_average = np.repeat(average, data.shape[1], axis=1)
        return constant_average
    else:
        raise ValueError("Invalid filtering method. Choose one of: 'moving_average', 'ema', 'lowpass', 'median', 'wavelet'")
    
def plot_results(ground_truths, ranked_results, video_idx):
    num_dims = 2

    fig, axs = plt.subplots(num_dims, 1, figsize=(24, 12), sharex=True)

    # Plot the first dimension of ground truth
    axs[0].plot(ground_truths[video_idx, :, 0], label='Ground Truth (Dim 1)')
    axs[1].plot(ground_truths[video_idx, :, 1], label='Ground Truth (Dim 2)')

    # Plot the predictions for each method
    for i, ((method, params), (score_result, interpolated_data)) in enumerate(ranked_results[:5]):
        axs[0].plot(interpolated_data[video_idx, :, 0], label=f'{method}, {params}')
        axs[1].plot(interpolated_data[video_idx, :, 1], label=f'{method}, {params}')

    axs[0].set_title(f'Video {video_idx} - Dimension 1')
    axs[1].set_title(f'Video {video_idx} - Dimension 2')

    axs[1].set_xlabel('Frame Index')
    for ax in axs:
        ax.legend()
    
    plt.show()

# Load the data
outputs_path = 'outputs/'
labeled_path = 'labeled/'

# Load the predictions
predictions = np.array([np.loadtxt(outputs_path + str(i+5) + '.txt') for i in range(5)])

# Load the ground truth labels
ground_truths = np.array([np.loadtxt(labeled_path + str(i) + '.txt') for i in range(5)])
# Define filtering methods and their parameter grids
filter_methods = ['base', 'moving_average', 'ema', 'lowpass', 'median', 'wavelet', 'savgol', 'constant_average', 'median400_850']
parameter_grids = {
    'base': {},
    'moving_average': {'window_size': [5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700]},
    'moving_average': {'window_size': [400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]},
    'ema': {'alpha': [0.001, 0.01, 0.03, 0.05, 0.1, 0.2]},
    'lowpass': {'cutoff_frequency': [0.01, 0.02, 0.03, 0.04, 0.05], 'order': [1, 2, 3, 4, 5]},
    'median': {'window_size': [5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700]},
    'median': {'window_size': [400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]},
    'wavelet': {'wavelet': ['db2', 'db4', 'db6'], 'level': [2, 6, 8], 'threshold': [0.05, 0.3, 1]},
    'kalman': {'process_variance': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2], 'measurement_variance': [1, 5, 10, 20, 50, 100, 200, 1000]},
    'savgol': {'window_size': [20, 40, 60, 80, 100, 200], 'order': [1, 2, 5, 9]},
    'average_methods': {'avg_size': [3, 5, 10, 20]},
    'constant_average': {},
    'median400_850': {}
}

# Initialize a dictionary to store results
results = {}

# Iterate over filter methods and parameter grids
for method in filter_methods:
    parameter_grid = list(ParameterGrid(parameter_grids[method]))

    for params in parameter_grid:
        print(method, params)
        # Apply the filter to the predictions
        filtered_predictions = filter_time_series(predictions, method=method, **params)

        # Evaluate the filtered predictions
        interpolated_data = interpolate_to_original_size(1200, filtered_predictions, axis=1)
        assert interpolated_data.shape == (5, 1200,2), (interpolated_data.shape, predictions.shape)
        score_result = score(interpolated_data)
        # Store the results
        results[(method, frozenset(params.items()))] = (score_result, interpolated_data)

# Rank the results
ranked_results = sorted(results.items(), key=lambda x: x[1][0])

# Display the ranked results
for i, ((method, params), (score_result, interpolated_data)) in enumerate(ranked_results):
    print(f"Rank {i+1}: Method={method}, Parameters={dict(params)}, Score={score_result}")


# Plot results for each video
for video_idx in range(5):
    plot_results(ground_truths, ranked_results, video_idx)