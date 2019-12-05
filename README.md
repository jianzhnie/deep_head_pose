# deep-head-pose

该项目是头部朝向模型的训练和测试

## 文件结构
- checkpoints : 模型训练过程中保存的文件
- datasets
    - 数据处理过程中用到的一些代码
    - filenamelists: 300W_LP和AFLW2000的文件名列表文件，一般用disgard版本，里面舍弃了角度在-99~99度以外的数据(论文的做法)
- models
    - 模型参数文件，包含了已经训练好的一些模型，hopenet*.pkl的模型是官方发布的模型，resnet50_epoch_4.pkl是我们重新训练的最好的模型，
    测试结果上，也是这四个模型中最优的
- network
    - HopeNet 网络定义
- test
    - 不同的测试方法： 1. 在公开数据集上的测试
                    2. 在video上的测试
                    3. 调用 detector 进行多人脸测试
- main.py : 训练的主文件，加入了分布式，训练参数可以参考代码文件
- tester.py :测试代码，测试参数可以参考代码文件
- train_test.sh: 训练、测试脚本，可根据需要修改该脚本的参数来进行训练测试

## 环境依赖
- pytorch 1.0

## 运行
运行`train_test.sh`脚本即可