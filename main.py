# 多py文件协同区
import pandas as pd

import ChangePathIntoCSV
import FeatureExtraction
import SetMyDataSet
import StartBp
import Start_SVM

# 宏定义区，用于定义各个功能函数的开关，True使能，FALSE 去使能
global change_path_into_CSV_switch                                        # 将路径汇总成一个CSV文件要不要执行
change_into_CSV_switch = False                                            # 只执行一遍，执行完了就把它关了，之后的所有switch起到一样的作用
global Feature_Extraction_switch                                          # 提取音频特征的代码要不要执行
Feature_Extraction_switch= False                                            # 特征提取要进行足足一个多小时
global Set_My_Data_Set_switch                                             # 要不要转换成数据库
Set_My_Data_Set_switch = True
global Start_Bp_Network_switch                                            # 要不要开始网络的训练
Start_Bp_Network_switch = False
global Start_SVM_switch                                                   # 要不要使用SVM进行训练和学习
Start_SVM_switch= True


# 定义显示格式
pd.set_option('display.max_columns', None)   # 显示完整的列
pd.set_option('display.max_rows', None)  # 显示完整的行
pd.set_option('display.expand_frame_repr', False)  # 设置不折叠数据
pd.set_option('display.max_colwidth', 100)

# 用于将MAV文件路径转换为CSV文件
if change_into_CSV_switch:
    ChangePathIntoCSV.change_into_CSV("09CSV")
# 提取音频特征,慎重执行。。。。。。。
if Feature_Extraction_switch:
    FeatureExtraction.feature_extraction("5FuaturesCSV",False,"09CSV")              #False表示不把文件名打印出来


if Set_My_Data_Set_switch:
    database, labelbase =SetMyDataSet.load_data("G:/dsp/5FuaturesCSV1.csv")
    print(database,labelbase)
if Start_Bp_Network_switch:
    StartBp.Start_Bp(data=database,label=labelbase, lr=0.02, epochs=3000 , n_feature= 172, n_hidden= 150 , n_output=10 )
if Start_SVM_switch:
    Start_SVM.Start_SVM(data=database,label=labelbase)

# 标识结束，显示各开关状态
print(" Program Finished\n", "change_into_CSV_switch:", change_into_CSV_switch,"\n Feature_Extraction_switch:",
      Feature_Extraction_switch, "\n Set_My_Data_Set_switch:", Set_My_Data_Set_switch )