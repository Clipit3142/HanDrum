import numpy as np, soundfile as sf, sounddevice as sd
import queue
import numpy
import json
import time

SR = 44100


class Sound:
    def __init__(self, path:str, n:int):
        try:
            self.vars = [self._load(f"{path}{i}.wav") for i in range(0,n)]
            self.n = n
        except FileNotFoundError:
            print("Error: file does not exist.")
        except json.JSONDecodeError:
            print("Error: file contains invalid JSON formatting.")

    def play(self, mixer:"Mixer", vol=1):
        mixer.play(self.vars[numpy.random.randint(0,self.n)])

    def _load(self, path:str):
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != SR:
            raise ValueError(f"{path}: {sr} Hz, expected {SR}")
        return data.mean(axis=1)

class Kit:
    def __init__(self, mixer, kit:str):
        try:
            with open(f"audio/{kit}/info.json", "r", encoding="utf-8") as file:
                self.d = json.load(file)
                self.kit = kit
        except FileNotFoundError:
            print("Error: file does not exist.")
        except json.JSONDecodeError:
            print("Error: file contains invalid JSON formatting.")

        self.sounds = {sound:Sound(f"audio/{kit}/{sound}/", self.d[sound]) for sound in self.d}
        self.slist = [sound for sound in self.d]
        self.mixer = mixer

    def play(self, sound:str):
        try:
            self.sounds[sound].play(self.mixer)
        except:
            print("sound not found")
        
class Mixer:
    def __init__(self, sr=SR, blocksize=256):
        self.voices = []
        self.incoming = queue.SimpleQueue()
        self.stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32",
                                      blocksize=blocksize, callback=self._cb)
        self.stream.start()

    def _cb(self, outdata, frames, time_info, status):
        while True:
            try: self.voices.append(self.incoming.get_nowait())
            except queue.Empty: break
        out = np.zeros(frames, dtype=np.float32)
        for v in self.voices:
            chunk = v[0][v[1]:v[1] + frames]
            out[:len(chunk)] += chunk
            v[1] += len(chunk)
        self.voices = [v for v in self.voices if v[1] < len(v[0])]
        outdata[:, 0] = np.clip(out, -1.0, 1.0)

    def play(self, samples:Sound):
        self.incoming.put([samples, 0])

    def close(self):
        self.stream.stop(); self.stream.close()