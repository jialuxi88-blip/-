# -
可以学习任意两种类型图片，例如猫狗、蚂蚁蜜蜂、古建筑和现代建筑等类似的，模型继承的pytorch resnet18

使用方法：

1.准备两种数据集图片，测试集每份120-160张左右，验证集每份60张左右，图片的文件目录是上传的dataset文件夹目录，不要更改位置！！！不要更改，文件夹名字只改'dataset→train→这里可以改'以及'dataset→val→这里可以改'

2.代码文件有两个，一个是训练模型的代码，一个是测试的代码，先用训练代码，再用测试代码

3.注意替换名称： ↓
# 标签映射
        self.label_map = {'classical_image': 0, 'morden_image': 1}
在运行代码.py文件中第50行# 标签映射中↑，替换自己分类的类型，'classical_image'和'morden_image'都必须替换成自己的dataset文件夹里的train下面的两个文件夹名字，名字与步骤1对应，注意是在代码中所有出现的这两种字符都要替换！！

在测试文件.py中，第49行'classical_architecture'，'morden_architecture'这两个字符串替换成需要分类的类别

4.测试文件的路径：文件夹名称是images

在测试文件.py中，第26行↓
IMAGE_PATH = 'images/2.png'，把2替换成自己的图片名称

5.准备完成，运行两个.py文件
