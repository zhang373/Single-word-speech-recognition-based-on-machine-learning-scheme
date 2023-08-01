import os
import csv
import pandas as pd
# fuck!!!! all the path cannot contain chinese
def change_into_CSV(file_name):
    digit = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    d = {}
    with open("G:/dsp/"+file_name+".csv", 'w') as csvfile:    # 打开对应文件夹创建了一个
        csvwriter = csv.writer(csvfile)                                                         # 创建一个csv的writer对象
        csvwriter.writerow(["File", "Label"])                                                   # 将一个list写成一行
        for x in digit:
            if os.path.isdir('G:/dsp/0_9resource/' + x):            # 分析0-9resource/x，x为0-9英文，是否为目录
                d[x] = os.listdir('G:/dsp/0_9resource/' + x)        # windows下的ls
                for name in os.listdir('G:/dsp/0_9resource/' + x):
                    if os.path.isfile('G:/dsp/0_9resource/' + x + "/" + name):  # 是否为文件
                        temp='G:/dsp/0_9resource/'+str(x) + '/' + str(name)
                        #print(temp)
                        csvwriter.writerow([temp, x])     # 改名并存一下，写在CSV里边
    df = pd.read_csv("G:/dsp/"+file_name+".csv")
    df = df.sample(frac=1)                                                                      # frac表示抽样数据量占比
    df.to_csv("G:/dsp/"+file_name+".csv", index=False)
    print(df.shape)
    return
