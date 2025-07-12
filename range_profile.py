import numpy as np
import matplotlib.pyplot as plt
from acconeer.exptool import a121

def main():
    # Connect to radar via UART port (change as needed)
    client = a121.Client.open(serial_port="/dev/ttyUSB0")
    print("Connected to server:", client.server_info)

    # Configure sensor parameters
    sensor_config = a121.SensorConfig()
    sensor_config.profile = a121.Profile.PROFILE_2
    sensor_config.start_point = 80       # adjust start range bin
    sensor_config.num_points = 100       # number of bins
    sensor_config.step_length = 1        # sampling step
    sensor_config.hwaas = 32             # hardware accelerated averaging samples
    sensor_config.sweeps_per_frame = 1

    # Setup and start the session
    client.setup_session(sensor_config)
    client.start_session()

    # Prepare plot
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_xlim(0, sensor_config.num_points)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Range bin")
    ax.set_ylabel("Amplitude")
    ax.set_title("Live Range Profile")

    try:
        while True:
            result = client.get_next()
            sweep_complex = result.frame[0, :]          # complex IQ data
            sweep_magnitude = np.abs(sweep_complex)    # convert complex to magnitude

            line.set_ydata(sweep_magnitude)
            line.set_xdata(np.arange(len(sweep_magnitude)))
            ax.set_ylim(0, np.max(sweep_magnitude) * 1.2)

            plt.pause(0.01)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        client.stop_session()
        client.close()


if __name__ == "__main__":
    main()
