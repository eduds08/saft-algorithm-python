import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import math

# Load the .mat file containing the data
file = "DadosEnsaio.mat"
mat = loadmat(file)

# Define the initial and final samples of the A-Scan
initial_sample = 500
final_sample = 900

# Extract B-Scan data (A-Scan values between the specified samples)
b_scan_data = mat["ptAco40dB_1"]["AscanValues"][0][0][initial_sample:final_sample]

# Extract the speed of sound in the material
sound_speed = mat["ptAco40dB_1"]["CscanData"][0][0]["Cl"][0][0][0][0]

# Extract the timescale
time_scale = mat["ptAco40dB_1"]["timeScale"][0][0][initial_sample:final_sample] * 1e-6

# Calculate the sampling period
sampling_period = time_scale[1][0] - time_scale[0][0]

# Convert time to position (accounting for round trip by dividing by 2)
depth_positions = sound_speed * time_scale / 2

# Extract the positions of the transducers
transducer_positions = mat["ptAco40dB_1"]["CscanData"][0][0]["X"][0][0] * 1e-3

# Display the B-Scan image
plt.figure()
plt.imshow(b_scan_data, aspect="auto")
plt.title('B-Scan')
plt.show()


def saft(b_scan_data, transducer_positions, depth_positions, sound_speed, sampling_period):
    # Initialize arrays for delays and focused image
    delays = np.zeros_like(b_scan_data, dtype=np.int64)
    focused_image = np.zeros_like(b_scan_data)

    # Loop through each transducer
    for transducer_idx in range(transducer_positions.size):
        # Loop through each depth position
        for depth_idx in range(depth_positions.size):
            # Loop through each lateral position
            for lateral_idx in range(transducer_positions.size):
                # Calculate the distance from the transducer to the focal point
                distance = math.sqrt(
                    depth_positions[depth_idx, 0]**2 +
                    (transducer_positions[lateral_idx, 0] - transducer_positions[transducer_idx, 0])**2
                ) - depth_positions[0, 0]

                # Calculate the delay in terms of sample indices
                delays[depth_idx, lateral_idx] = round(distance * 2 / sound_speed / sampling_period)

                # Accumulate the signal if the delay is within valid range
                if delays[depth_idx, lateral_idx] < len(b_scan_data):
                    focused_image[depth_idx, lateral_idx] += b_scan_data[delays[depth_idx, lateral_idx], transducer_idx]

    return focused_image


# Apply SAFT to the B-Scan data
focused_image = saft(b_scan_data, transducer_positions, depth_positions, sound_speed, sampling_period)

# Display the focused image
plt.figure()
plt.imshow(focused_image, aspect='auto')
plt.title('SAFT')
plt.show()
