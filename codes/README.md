采用cnn对mnist数据集进行分类

实现了卷积及平均池画的中正向传播与反向传播。实现了交叉熵损失函数。

可在output_paths.py中修改记录loss值的文件路径。

run_cnn.py中save_parameters设为true会保存所有的W矩阵和b向量的值，use_parameters可使用保存的parameters继续训练。

visualization.py用于可视化，会分别可视化一张图片在1、10、100个epochs后经过一、二层卷积的输出结果。

plot用于绘制loss曲线。将plot_single设为true会只绘制cnn的loss曲线，否则绘制cnn和mlp两者的（cnn的loss记录文件应为output_cnn.csv，mlp的应为output_mlp.csv）。
