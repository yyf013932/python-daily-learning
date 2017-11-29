import wave
from scipy.io import wavfile
import matplotlib.pyplot as plt


def read_wav(filename):
    # 采样率、数据
    freq, sda = wavfile.read(filename)
    # 数据为np array
    print(type(sda))
    # 根据单声道和双声道，可能为n*1 和 n*2大小
    print(sda.shape)
    if sda.ndim == 1:
        x = sda.size
        y = sda
    else:
        x = sda.shape[0]
        y = sda[:, 0]
    # 做出声波图
    plt.plot(x, y)


'''
输出指定格式的wav文件
sample_freq:采样频率
wav_data:声音数据，必须是n*1（单声道）或n*2（立体声）
sample_width:音频位数，以byte为单位，如sample_width为2表示为16位
file_name:存储路径
'''


def write_wav(sample_freq, wav_data, sample_width, file_name):
    f = wave.open(file_name, "wb")
    if wav_data.ndim > 2:
        return
    if wav_data.ndim == 1 or wav_data.shape[1] == 1:
        channels = 1
    else:
        channels = 2
    # set wav params
    f.setnchannels(channels)
    f.setsampwidth(sample_width)
    f.setframerate(sample_freq)
    # turn the data to string
    f.writeframes(wav_data.tostring())
    f.close()
