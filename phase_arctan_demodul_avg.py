import numpy as np
import matplotlib.pyplot as plt
from acconeer.exptool import a121
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.fft import fft, fftfreq


def compute_fft(signal, fs):
    n = len(signal)
    freq = fftfreq(n, d=1/fs)
    spectrum = np.abs(fft(signal)) / n
    return freq[:n//2], spectrum[:n//2]  # Return only the positive frequencies

def bandpass_filter(signal, low, high, fs, order=3):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)

def main():
    # Connect to radar over UART (USB)
    client = a121.Client.open(serial_port="/dev/ttyUSB0")

    # Config params
    frame_rate = 20  # Hz
    duration = 10 
       # seconds
    num_frames = frame_rate * duration
    sweeps_per_frame = 1
    num_points = 100

    # Sensor config
    config = a121.SensorConfig(
        profile=a121.Profile.PROFILE_3,
        frame_rate=frame_rate,
        start_point=50,
        num_points=num_points,
        step_length=1,
        sweeps_per_frame=sweeps_per_frame,
    )

    # Setup and start
    client.setup_session(config)
    client.start_session()

    # Initialize
    iq_data = np.zeros((num_frames, sweeps_per_frame, num_points), dtype=complex)
    phase_series = []
    print("collecting data")
    for i in range(num_frames):
        result = client.get_next()
        iq_data[i] = result.frame  # shape: (sweeps_per_frame, num_points)

        # Get phase at various points in a sweep
        phase = np.angle(result.frame[0])  # shape: (num_points,)
        

        # Mean of phase at one sweep or 1 frame here
        mean_phase=np.mean(phase)
        phase_series.append(mean_phase)
    
    
    print("done recording")
    client.stop_session()
    client.close()

    # Convert phase list to numpy array
    phase_series = np.array(phase_series)

    # Unwrap phase to remove 2π jumps
    unwrapped_phase = np.unwrap(phase_series)
      # Plot
    time_axis = np.linspace(0, duration,(duration*frame_rate))
    plt.plot(time_axis,unwrapped_phase)
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (radians)")
    plt.title("Phase arc tangent demodulation tracking max amplitude range bin")
    plt.grid(True)
    plt.show()
if  __name__== "__main__":
    main()
           