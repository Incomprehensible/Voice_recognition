import librosa as voice
import numpy as magic

# версия для простых, четких парней

# частота дискретизации

FR = 16000

def filter_smooth(apsum):
    apsum -= magic.min(apsum)
    apsum /= magic.max(apsum)
    return apsum

def filter_audio(audio):
    voicepwr = voice.amplitude_to_db(magic.abs(voice.stft(audio, n_fft=2048)), ref=magic.max)
    apsum = filter_smooth(magic.sum(voicepwr, axis=0) ** 2)
    apsum = filter_smooth(magic.convolve(apsum, magic.ones((9,)), 'same'))
    apsum = magic.array(apsum > 0.35, dtype=bool)
    apsum = magic.repeat(apsum, magic.ceil(len(audio) / len(apsum)))[:len(audio)]
    return audio[apsum]

def process_audio(afile):
    audio, _ = voice.load(afile, sr=FR)
    afs = voice.feature.mfcc(filter_audio(audio), sr=FR, n_mfcc=34, n_fft=2048)

    afs_sum = magic.sum(afs[2:], axis=-1)
    afs_sum = afs_sum / magic.max(afs_sum)
    return afs_sum

def confidence(x, y):
    return magic.sum((x - y) ** 2)

man1 = process_audio('man1.wav')
man12 = process_audio('man1.2.wav') # same as man1
man2 = process_audio('man2.wav')
woman1 = process_audio('woman1.wav')
woman2 = process_audio('woman2.wav')
print('same', confidence(man1, man12))
print('diff', confidence(man1, man2))
print('diff', confidence(woman1, woman2))
