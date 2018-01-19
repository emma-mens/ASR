import os
import sys
import subprocess
import numpy as np
import random
import yaml

class GetData:
  def __init__(self, audio_dir=None, freq_bins=1025, samples_per_frame=20, train_frame_shift=60, test_frame_shift=1, spectrogram_dir='SPECTROGRAM_DIR'):

    self.AUDIO_DIR          = '/home/emazuh/MedleyDB/Audio/' if not audio_dir else audio_dir
    self.MIXTURE_MAG_SUFFIX = '_MIX_mag.txt'
    self.YAML_SUFFIX        = '_METADATA.yaml'
    self.STEM_SUFFIX        = '_STEMS'
    self.SPEC_DIR           = spectrogram_dir # directory to store stem spectrograms relative to dir in self.audio_dirs/**STEM/
    self.NUM_FREQ_BINS      = freq_bins
    self.SAMPLES_PER_FRAME  = samples_per_frame
    self.TRAIN_FRAME_SHIFT  = train_frame_shift
    self.TEST_FRAME_SHIFT   = test_frame_shift
    self.audio_dirs         = os.listdir(self.AUDIO_DIR)
    self.NUM_SONGS          = len(self.audio_dirs)

    #random.shuffle(self.audio_dirs)
    #split = int(0.7*self.NUM_SONGS) # split training and test sets into 70, 30 percentage

    self.train_audio_dirs   = self.audio_dirs
    self.test_audio_dirs    = ['TheDistricts_Vermont']# self.audio_dirs[split:]
    self.NUM_TRAIN_SONGS    = len(self.train_audio_dirs)
    self.NUM_TEST_SONGS     = len(self.test_audio_dirs)
    self.next_song_idx      = 0 # Feed data per new song directory for batch training

  def get_next_train_data(self):
  
    # Returns training set input and labels for one song directory. This is 
    # used during batch training, where each batch is a set of all frames for one song
  
    audio_dir = self.train_audio_dirs[self.next_song_idx]
    self.next_song_idx += 1

    # Convention for naming yaml file and stem directories implemented by MedleyDB
    base = os.path.join(self.AUDIO_DIR, audio_dir, audio_dir)
    yaml_file = base + self.YAML_SUFFIX
    stem_dir = base + self.STEM_SUFFIX

    # My naming convention for the mixture spectrogram magnitude file
    spectrogram_mag_file = base + self.MIXTURE_MAG_SUFFIX
  
    return self.get_input_data(spectrogram_mag_file, self.TRAIN_FRAME_SHIFT), self.get_labels(stem_dir, yaml_file, self.TRAIN_FRAME_SHIFT)

  def get_test_data(self):
    
    # Returns all test input and labels
    data = np.empty((1, self.NUM_FREQ_BINS, self.SAMPLES_PER_FRAME))
    labels = np.empty((1, self.NUM_FREQ_BINS, self.SAMPLES_PER_FRAME))

    for audio_dir in self.test_audio_dirs:
      base = os.path.join(self.AUDIO_DIR, '../test/', audio_dir, audio_dir)
      yaml_file = base + self.YAML_SUFFIX
      stem_dir = base + self.STEM_SUFFIX

      spectrogram_mag_file = base + self.MIXTURE_MAG_SUFFIX
      data = np.vstack([data, self.get_input_data(spectrogram_mag_file, self.TEST_FRAME_SHIFT)])
      labels = np.vstack([labels, self.get_labels(stem_dir, yaml_file, self.TEST_FRAME_SHIFT)])

    return data, labels
  
  def get_input_data(self, file_name, frame_shift):

    arr = self.specfile_to_array(file_name)
    arr = arr/float(np.max(np.abs(arr))) # Peak normalize
    # arr -= np.mean(arr) # Mean centering

    # Make input data: SAMPLES_PER_FRAMES context frames per input, skip frame_shift frames
    N_FREQ_BINS, N_SAMPLES = arr.shape
    INPUT_SIZE = (N_SAMPLES - self.SAMPLES_PER_FRAME)//frame_shift 
    input_samples = np.zeros((INPUT_SIZE, N_FREQ_BINS, self.SAMPLES_PER_FRAME))
    
    for i in xrange(INPUT_SIZE):
        input_samples[i, :, :] = arr[:, i*frame_shift : i*frame_shift + self.SAMPLES_PER_FRAME]
  
    return input_samples

  def specfile_to_array(self, file_name):
      with open(file_name, 'r') as f:
          data = f.readlines()
      # Convert text spectrogram information into numpy 2D array [[freq_bins] time]
      return np.array(map(lambda line: map(float, line.split('\n')[0].split(',')), data))
  
  def get_labels(self, directory, yaml_file, frame_shift):
      # Labels 1 for vocals and 0 for non vocals as the ideal binary mask, comparing magnitude of vocal/nov-vocal spectrogram
      vocals, non_vocals = self.get_music_types(yaml_file)
      self.convert_wav_to_csv(directory, vocals + non_vocals)
  
      if len(vocals) == 0:
          return np.zeros(self.get_input_data(self.get_spec_file_name(directory, non_vocals[0]), frame_shift).shape) 
      if len(non_vocals) == 0:
          return np.ones(self.get_input_data(self.get_spec_file_name(directory, vocals[0]), frame_shift).shape)

      v_sum = np.add.reduce([self.get_input_data(self.get_spec_file_name(directory, f), frame_shift) for f in vocals])*1.0/len(vocals)
      n_sum = np.add.reduce([self.get_input_data(self.get_spec_file_name(directory, f), frame_shift) for f in non_vocals])*1.0/len(non_vocals)
      return np.greater(v_sum, n_sum).astype(int)
  
  def convert_all_wav_files(self):
      for audio_dir in os.listdir(self.AUDIO_DIR):
      
          directory = os.path.join(self.AUDIO_DIR, audio_dir)
          yaml_file = os.path.join(directory, audio_dir + '_METADATA.yaml')
          stem_dir = os.path.join(directory, audio_dir + '_STEMS')
          mix_wav = os.path.join(directory, audio_dir + '_MIX.wav')
          out_mix = os.path.join(directory, audio_dir + '_MIX')

          # Make sure all required files and directories exist
          assert os.path.exists(yaml_file)
          assert os.path.exists(stem_dir)
          assert os.path.exists(mix_wav)
          
          vocals_wavs, nonvocal_wavs = self.get_music_types(yaml_file)
          vocals_wavs.extend(nonvocal_wavs)
          
          # Convert MIX file to spectrogram
          # TODO: remove matlab dependency. replace with matplotlib.pyplot.specgram
          if os.path.exists('get_spectrogram.m') and not os.path.exists(out_mix + '_mag.txt'):
              matlab_cmd = 'matlab -nosplash -nodisplay -r \"get_spectrogram(\'%s\',\'%s\')\"'
              subprocess.call([matlab_cmd % (mix_wav, out_mix)], shell=True)
          
          self.convert_wav_to_csv(stem_dir, vocals_wavs)
  
  def convert_wav_to_csv(self, directory, wavfiles):
      # wavfiles is a list wavfile names in directory

      target_directory = os.path.join(directory, self.SPEC_DIR) # store output of get_spectrogram.m here
      if not os.path.exists(target_directory):
          os.mkdir(target_directory)
      
      in_files = map(lambda f: os.path.join(directory, f), wavfiles)
      out_files = map(lambda f: os.path.join(directory, target_directory, f.split('.')[0]), wavfiles)
          
      if os.path.exists('get_spectrogram.m'):
          matlab_cmd = 'matlab -nosplash -nodisplay -r \"get_spectrogram(\'%s\',\'%s\')\"'
          for i in xrange(len(wavfiles)):
              if not os.path.exists(self.get_spec_file_name(directory, out_files[i])):
                  subprocess.call([matlab_cmd % (in_files[i], out_files[i])], shell=True)
      else:
          print 'make sure you have get_spectrogram.m in this directory'
          exit()
  
  def get_spec_file_name(self, directory, wavfile):
      return os.path.join(directory, self.SPEC_DIR, wavfile.split('.')[0] + '_mag.txt')
  
  
  def get_music_types(self, yaml_file):
      # yaml_file contains info on wavfiles
      # Returns vocal, non_vocal (both lists of vocal files and non vocal files)

      with open(yaml_file, 'r') as y:
          d = yaml.load(y)
      parts = d['stems']
      vocals = filter(lambda stem: parts[stem]['instrument'].find('male singer') != -1, parts)
      vocals = map(lambda stem: parts[stem]['filename'], vocals)
      
      non_vocals = filter(lambda stem: parts[stem]['instrument'].find('male singer') == -1, parts)
      non_vocals = map(lambda stem: parts[stem]['filename'], non_vocals)
      
      return vocals, non_vocals