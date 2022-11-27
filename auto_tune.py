from functools import partial
from pathlib import Path
import argparse
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as sig
import psola


SEMITONES_IN_OCTAVE = 12


def closest_pitch(f0):
    nan_indices = np.isnan(f0)
    midi_note = np.around(librosa.hz_to_midi(f0))
    midi_note[nan_indices] = np.nan
    return librosa.midi_to_hz(midi_note)


def closest_sanitized_pitch(f0, scale):
    if np.isnan(f0):
        return np.nan
    degrees = degrees_from(scale)
    midi_note = librosa.hz_to_midi(f0)
    degree = midi_note % SEMITONES_IN_OCTAVE
    degree_id = np.argmin(np.abs(degrees - degree))
    degree_difference = degree - degrees[degree_id]
    midi_note -= degree_difference
    return librosa.midi_to_hz(midi_note)


def degrees_from(scale: str):
    degrees = librosa.key_to_degrees(scale)
    degrees = np.concatenate((degrees, [degrees[0] + SEMITONES_IN_OCTAVE]))
    return degrees


def aclosest_sanitized_pitch(f0, scale):
    sanitized_pitch = np.zeros_like(f0)
    for i in np.arange(f0.shape[0]):
        sanitized_pitch[i] = closest_sanitized_pitch(f0[i], scale)
    smoothed_sanitized_pitch = sig.medfilt(sanitized_pitch, kernel_size=11)
    smoothed_sanitized_pitch[np.isnan(smoothed_sanitized_pitch)] = \
        f0[np.isnan(smoothed_sanitized_pitch)]
    return smoothed_sanitized_pitch


def autotune(audio, sr, correction_function, plot=False):
    frame_length = 2048
    hop_length = frame_length // 4
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')
    
    f0, voiced_flag, voiced_probabilities = librosa.pyin(audio,
                                                         frame_length=frame_length,
                                                         hop_length=hop_length,
                                                         sr=sr,
                                                         fmin=fmin,
                                                         fmax=fmax)

    corrected_f0 = correction_function(f0)

    if plot:
        time_points = librosa.times_like(f0)
        stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
        log_stft = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(log_stft, x_axis='time', y_axis='log', ax=ax)
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        ax.plot(time_points, f0, label='original pitch', color='cyan', linewidth=2)
        ax.plot(time_points, corrected_f0, label='corrected pitch', color='orange', linewidth=1)
        ax.legend(loc='upper right')
        plt.savefig('pitch_correction.png', dpi=300, bbox_inches='tight')

    return psola.vocode(audio, sample_rate=int(sr), target_pitch=corrected_f0, fmin=fmin, fmax=fmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('vocals_file')
    ap.add_argument('--plot', '-p', action='store_true', default=False,
                    help='if set, will produce a plot of the results')
    ap.add_argument('--correction-method', '-c', choices=['closest', 'scale'], default='closest')
    ap.add_argument('--scale', '-s', type=str, help='see librosa.key_to_degrees;'
                                                    ' used only for the \"scale\" correction'
                                                    ' method')
    args = ap.parse_args()
    
    filepath = Path(args.vocals_file)
    
    y, sr = librosa.load(str(filepath), sr=None, mono=False)
    if y.ndim > 1:
        y = y[0, :]

    correction_function = closest_pitch if args.correction_method == 'closest' else \
        partial(aclosest_sanitized_pitch, scale=args.scale)
    
    pitch_corrected_y = autotune(y, sr, correction_function, args.plot)
    
    filepath = filepath.parent / (filepath.stem + '_pitch_corrected' + filepath.suffix)
    sf.write(str(filepath), pitch_corrected_y, sr)

    
if __name__=='__main__':
    main()
    