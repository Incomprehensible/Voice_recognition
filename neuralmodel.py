import librosa as lr
import numpy as magic

from keras.layers import Dense, LSTM, Activation
from keras.models import Sequential
from keras.optimizers import Adam

FR = 16000
LEN = 16
OVERLAP = 8
FFT = 1024

def filter_smooth(apsum):
    apsum -= magic.min(apsum)
    apsum /= magic.max(apsum)
    return apsum

def filter_audio(audio):
    voicepwr = lr.amplitude_to_db(magic.abs(lr.stft(audio, n_fft=2048)), ref=magic.max)
    apsum = filter_smooth(magic.sum(voicepwr, axis=0) ** 2)
    apsum = filter_smooth(magic.convolve(apsum, magic.ones((9,)), 'same'))
    apsum = magic.array(apsum > 0.35, dtype=bool)
    apsum = magic.repeat(apsum, magic.ceil(len(audio) / len(apsum)))[:len(audio)]
    return audio[apsum]

def audio_get_ready(afile, target=False):
    print("loading data %s" % afile)
    audio, _ = lr.load(afile, sr=FR)
    data = lr.stft(filter_audio(audio), n_fft=FFT).swapaxes(0, 1)
    samples = []

    for i in range(0, len(data) - LEN, OVERLAP):
        samples.append(magic.abs(data[i:i + LEN]))

    results_shape = (len(samples), 1)
    results = magic.ones(results_shape) if target else magic.zeros(results_shape)
    return magic.array(samples), results

voices = [("woman2.wav", True),
          ("woman1.wav", False),
          ("man1.wav", False),
          ("man1.2.wav", False),
          ("man2.wav", False)]
X, Y = audio_get_ready(*voices[0])
for voice in voices[1:]:
    dx, dy = audio_get_ready(*voice)
    X = magic.concatenate((X, dx), axis=0)
    Y = magic.concatenate((Y, dy), axis=0)
    del dx, dy

perm = magic.random.permutation(len(X))
X = X[perm]
Y = Y[perm]

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=X.shape[1:]))
model.add(LSTM(64))
model.add(Dense(64))
model.add(Activation('tanh'))
model.add(Dense(16))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('hard_sigmoid'))

model.compile(Adam(lr=0.005), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, Y, epochs=15, batch_size=32, validation_split=0.2)

print(model.evaluate(X, Y))
model.save('model.newborn_hdf5')