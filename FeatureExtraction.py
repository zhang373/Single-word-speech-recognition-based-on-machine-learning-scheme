# print("1")
import librosa   # 这个东西要安装一下 第一遍弄得时候会报错，需要处理numba llvmlite的问题，是真的很恶心。。。。。。。。。。。。 我的建议是别搞
# print("yes")   #这个yes和1是用来判定librosa能不能正常使用的
import numpy as np
import pandas as pd
import csv
import os

# 一共用了五个状态信号作为音频的特征值 使用librosa提取出来的，提取出俩之后写在5FuaturesCSV（给定的名称）
# - Mel Frequency Cepstral Coefficients (MFCCs)：根据人类听觉系统的响应（Mel尺度）间隔的频带组成声音的频谱表示的系数。
# - Chroma：与12个不同的音高等级有关。
# - Mel spectrogram：它的平均值-基于Mel标度的Mel谱图。
# - Spectral Contrast：表示谱的质心。
# - Tonnetz：代表音调空间。

def get_data(files,Show,digit):
    pieces_batch= 10
    length = 50#len(files.File)-5                                         # 主动舍弃五行数据，防止超过边界！！！！！！妈的智障
    i = 0                                                                   # 表示大循环的脚标
    j = 0                                                                   # 用于标识分段的dataframe
    k = 0                                                                   # 在每一个batch里边进行操作，循环标识符
    # print(librosa.load(files.File[0], sr=22050))
    length1= len(librosa.load(files.File[0], sr=22050)[0])                  # data的行宽，data.shape中的参数
    data = np.empty((0, length1), int)                                      # 空data的DataFrame，用于动态扩张
    indexs = []                                                             # 存label的
    datalen = 0                                                             # 用于标识是否应该转换为.csv文件
    while i < length:                                                       # 所有文件的循环
        while k < pieces_batch and i < length:                              # 单一批次内操作，防止后边操作的太慢了，昨晚上弄一晚上没弄好
            temp = librosa.load(files.File[i], sr=22050)[0].T               # 对于单个音频进行采样，一共22050个采样点
            temp = np.array(temp)                                           # 转成array，用于实现扩张性
            if Show:
                print(type(temp),type(data))
                print(temp.shape,data.shape)
            temp=temp.reshape(temp.shape[0],1)                              # 列向量转成行向量
            temp=temp.T
            if Show:
                print(type(temp),type(data))
                print(temp.shape,data.shape)
            i += 1
            k += 1
            if temp.shape[1] == 22050:                                  # 有一些时长不够，采不够22050个点，删掉，只留下完整的
                data = np.append(data,np.array(temp), axis=0)
                indexs.append(str(digit.index((files.Label[i]))))
                datalen += 1
            if Show:
                print(i,datalen)
        k = 0
        if datalen != 0:                                                # 将处理好的一个批次留存在csv里边，减少堆栈压力，提速
            df = pd.DataFrame(data)
            df.to_csv("G:/dsp/" + str(j) + ".csv", index=False)
            j += 1
        datalen = 0
        data = np.empty((0, length1), int)                              # 清空data之后进入新的循环
    k = 0
    data = pd.read_csv("G:/dsp/" + str(k) + ".csv")                     # 开始读取存放的checkpoint
    k += 1
    while k < j:
        temp = pd.read_csv("G:/dsp/" + str(k) + ".csv")
        data = pd.concat([data, temp])                                  # 融合所有checkpoint的数据
        k += 1
    print(data)
    data = data.values
    indexs = np.array(indexs)                                           # 将indexs转置
    if Show:
        print(type(indexs))
        print(indexs.shape)
    indexs = indexs.reshape(indexs.shape[0],1)
    indexs = indexs.T
    if Show:
        print(type(indexs))
        print(indexs.shape)
    return data, indexs                                                 # 退出data和标签

# 识别语音的特征
def extract_features( files, ShowSwitch,filename):
    digit = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    # files之后传的参数就是ChangePathIntoCSV中的文件名称
    # print('E:/张文硕/大学/大三上半学期/DSP/语言识别/0-9resource/' + files.File)
    # print(str(files.File))
    data, indexs = get_data(files,True,digit)                      # 获取采样完成的数据和标签
    sr=22050                                                 # 每一段采样22050次
    if ShowSwitch:
        print( str(files.File))                              # 展示文件名称
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr).T, axis=0)                                              # 提取Mel Frequency Cepstral Coefficients
    # spectral_centroids = librosa.feature.spectral_centroid(data + 0.01, sr=sr)[0]
    stft = np.abs(librosa.stft(data))                                                                           # 取abs，用在下边提取chroma的
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)                                      # 与12个不同的音高等级有关
    mel = np.mean(librosa.feature.melspectrogram(data, sr).T, axis=0)                                           # 它的平均值-基于Mel标度的Mel谱图
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)                              # 谱的质心
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sr).T, axis=0)               # 代表音调空间
    if ShowSwitch:
        print(mfccs.shape, stft.shape, chroma.shape, mel.shape, contrast.shape, tonnetz.shape)

    row = np.concatenate((mfccs, chroma, mel, contrast, tonnetz), axis = 0).astype('float32')                   # 上边一共提取了五个特征值
    if ShowSwitch:
        print("row!!!\n",row)
        print(np.concatenate((indexs, row)),type(np.concatenate((indexs, row))),(np.concatenate((indexs, row))).shape)                                # 合并并存放
        print(pd.DataFrame(indexs).sample(frac=1), indexs.shape, pd.DataFrame(row).sample(frac=1),row.shape)
    print(pd.DataFrame(np.append(indexs, row, axis=0)).sample(frac=1))
    pd.DataFrame(np.append(indexs, row, axis=0)).T.to_csv("G:/dsp/"+filename+"1.csv",index=False, header= False)


def feature_extraction(filename, ShowSwitchOuter, Pathfilename):
    sp = pd.read_csv("G:/dsp/"+Pathfilename+".csv")                         #存放位置的地址
    # 最关键这个apply函数，sp读取了所有数据集路径，
    # apply(extract_features, axis=1)，把路径当做参数传进extract_features运行。
    extract_features(files=sp,ShowSwitch=ShowSwitchOuter,filename=filename)