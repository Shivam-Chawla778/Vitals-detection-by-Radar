import numpy as np
import matplotlib.pyplot as plt
from acconeer.exptool import a121

# Parameters
NUM_SWEEPS = 100             # Width of heatmap (x-axis, time)
SERIAL_PORT = "/dev/ttyUSB0" # Update this to your correct port

def main():
    # Connect to radar
    client = a121.Client.open(serial_port=SERIAL_PORT)
    print("Connected to server:", client.server_info)

    # Radar configuration
    sensor_config = a121.SensorConfig()
    sensor_config.profile = a121.Profile.PROFILE_2
    sensor_config.start_point = 80
    sensor_config.num_points = 100
    sensor_config.step_length = 1
    sensor_config.hwaas = 32
    sensor_config.sweeps_per_frame = 1

    # Setup radar session
    client.setup_session(sensor_config)
    client.start_session()

    # Create empty matrix: [num_range_bins x num_sweeps]
    num_bins = sensor_config.num_points
    range_matrix = np.zeros((num_bins, NUM_SWEEPS))

    # Initialize heatmap
    fig, ax = plt.subplots()
    img = ax.imshow(range_matrix,
                    aspect='auto',
                    origin='lower',
                    cmap='viridis',
                    interpolation='nearest',
                    extent=[0, NUM_SWEEPS, 0, num_bins])
    ax.set_xlabel("Time (sweeps)")
    ax.set_ylabel("Range bin")
    ax.set_title("Live Range-Slow-Time Matrix")

    plt.colorbar(img, ax=ax, label="Amplitude")

    try:
        while True:
            result = client.get_next()
            sweep = result.frame[0, :]                # complex data
            sweep_mag = np.abs(sweep)                # magnitude

            # Shift matrix left and append new column
            range_matrix = np.roll(range_matrix, -1, axis=1)
            range_matrix[:, -1] = sweep_mag

            # Update heatmap
            img.set_data(range_matrix)
            img.set_clim(0, np.max(range_matrix) * 1.1)  # adjust color scaling
            plt.pause(0.01)

    except KeyboardInterrupt:
        print("Interrupted, exiting.")
    finally:
        client.stop_session()
        client.close()

if __name__ == "__main__":
    main()
