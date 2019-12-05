import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置绘图风格
#plt.style.use('ggplot')
# 设置中文编码和负号的正常显示
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# 读取需要绘图的数据
df1=pd.read_csv('result/multinet/resnet18_2.0_train_result.csv',sep=',') 
df2=pd.read_csv('result/multinet/resnet18_2.0_test_result.csv',sep=',') 
df2.columns = ['epoch','yaw_test','pitch_test','roll_test','mae_test']
df = pd.merge(df1, df2, on = 'epoch')

label_list = ['yaw','pitch','roll','mae']

for direction in label_list:
	# 设置图框的大小
	fig = plt.figure(figsize=(5,3))
	# 绘图
	# epoch,yaw,pitch,roll,mae
	plt.plot(df.epoch, # x轴数据
			df[direction], # y轴数据
			linestyle = '-', # 折线类型
			linewidth = 2, # 折线宽度
			color = 'steelblue', # 折线颜色
			marker = '.', # 点的形状
			markersize = 2, # 点的大小
			markeredgecolor='black', # 点的边框色
			markerfacecolor='brown',
			label = 'train') # 点的填充色
	# 添加标题和坐标轴标签
	plt.title('Training Loss')
	plt.xlabel('epoch')
	plt.ylabel(direction)
	# 为了避免x轴日期刻度标签的重叠，设置x轴刻度自动展现，并且45度倾斜
	fig.autofmt_xdate(rotation = 45)
	plt.legend()
	plt.savefig('/home/dm/Desktop/resnet18/multinet/'+  direction + "2_train.png")
	
	fig = plt.figure(figsize=(5,3))
	# # 显示图形
	# plt.show()
	plt.plot(df.epoch, # x轴数据
			df[direction + '_test'], # y轴数据
			linestyle = '-', # 折线类型
			linewidth = 2, # 折线宽度
			color = '#ff9999', # 折线颜色
			marker = '.', # 点的形状
			markersize = 2, # 点的大小
			markeredgecolor='black', # 点的边框色
			markerfacecolor='brown',
			label='test') # 点的填充色
		# 添加标题和坐标轴标签
	plt.title('Test Error')
	plt.xlabel('epoch')
	plt.ylabel(direction)
	# 为了避免x轴日期刻度标签的重叠，设置x轴刻度自动展现，并且45度倾斜
	fig.autofmt_xdate(rotation = 45)
	plt.legend()
	plt.savefig('/home/dm/Desktop/resnet18/multinet/'+  direction + "2_test.png")
	# # 显示图形
	# plt.show()

