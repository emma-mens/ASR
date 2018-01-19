from scipy.signal import istft
import utils

fetch_data = utils.GetData()

x,y = fetch_data.get_test_data()


def audio_from_spec(phase_file, mag_spectrogram, fs=44100):
  phase = fetch_data.specfile_to_array(phase_file)
  return istft(np.multiply(mag_spectrogram, np.e**(1j*phase)), fs=fs)