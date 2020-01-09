import librosa as voice
import numpy as magic
import asyncio

# более удачная попытка ускорения программы засчет испольования асинхронных потоков

# частота дискретизации

FR = 16000
voices = []
naudios = magic.ndarray([])
vectors = magic.ndarray([])

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

def process_audio():
    global voices
    for item in vectors:
        afs = voice.feature.mfcc(filter_audio(item), sr=FR, n_mfcc=34, n_fft=2048)
        afs_sum = magic.sum(afs[2:], axis=-1)
        afs_sum = afs_sum / magic.max(afs_sum)
        voices.append(afs_sum)

def confidence(x, y):
    return magic.sum((x - y) ** 2)

async def load_file(afile, audios):
    audio, _ = voice.load(afile, sr=FR)
    audios.append(audio)

async def load_audio(files, audios):
    tasks = []
    for file in files:
        task = asyncio.ensure_future(load_file(file, audios))
        tasks.append(task)
    await asyncio.gather(*tasks, return_exceptions=True)

def print_confidence(array):
    print(confidence(array[0], array[1]))
    print(confidence(array[1], array[2]))
    print(confidence(array[2], array[3]))
    print(confidence(array[3], array[4]))

if __name__ == "__main__":
    files = ['man1.wav', 'man1.2.wav', 'man2.wav', 'woman1.wav', 'woman2.wav']
    audios = []
    asyncio.get_event_loop().run_until_complete(load_audio(files, audios))
    vectors = magic.asarray(audios)
    process_audio()
    naudios = magic.asarray(voices)
    print_confidence(naudios)
