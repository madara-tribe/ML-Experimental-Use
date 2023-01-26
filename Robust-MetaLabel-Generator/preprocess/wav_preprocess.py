import cv2, os
import matplotlib.pyplot as plt
import numpy as np

import wav_func as function
# STFT計算の関数
def stft(path, Fs, overlap):
    # wavファイルを読み込む(縦軸の校正はしていない)
    data, samplerate = function.wavload(path)
    data_ = data.reshape(224, 224).astype(np.float32)*255
    data_ = cv2.cvtColor(data_, cv2.COLOR_GRAY2RGB)
    print(data_.shape)
    #fft_array = cv2.cvtColor(fft_array, cv2.COLOR_GRAY2RGB)
    plt.imshow(data_),plt.show()
    # オーバーラップ抽出された時間波形配列
    time_array, N_ave, final_time = function.ov(data, samplerate, Fs, overlap)
    #print(time_array.shape)
    # ハニング窓関数をかける
    time_array, acf = function.hanning(time_array, Fs, N_ave)
    #print(time_array.shape)
    # FFTをかける
    fft_array, fft_mean, fft_axis = function.fft_ave(time_array, samplerate, Fs, N_ave, acf)
    #print(fft_array.shape)
    # スペクトログラムで縦軸周波数、横軸時間にするためにデータを転置
    fft_array = fft_array.T
    return fft_array, fft_axis, final_time, samplerate

if __name__=="__main__":
    wavpath = "sample.wav"
    Fs = 4096   # frame size while FFT
    overlap = 75 
    # STFT calculation
    fft_array, fft_axis, final_time, samplerate = stft(wavpath, Fs, overlap)
    print(fft_array.shape)
    #fft_array = fft_array.reshape(256, 240, 3)
    #fft_array = cv2.cvtColor(fft_array, cv2.COLOR_GRAY2RGB)
    #plt.imshow(fft_array, "gray"),plt.show()
