import argparse
from utils import GetData

parser = argparse.ArgumentParser(description='Create spectrogram magnitude files in dataset.')

parser.add_argument('--audio-dir', type=str, dest='audio_dir',
                    help='Directory of MedleyDB Audio files', required=True)

if __name__ == '__main__':

    d = GetData(audio_dir=audio_dir)
    d.convert_all_wav_files()