import torch
import torch.nn as nn

# 定义WrappedGPT类
class WrappedGPT:
    """
    这个类封装了一个GPT层,用于特定的操作。
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        # 存储传入的层
        self.layer = layer
        # 从层的权重中获取设备信息
        self.dev = self.layer.weight.device
        # 获取权重的行数（输出维度大小）
        self.rows = layer.weight.data.shape[0]
        # 获取权重的列数（输入维度大小）
        self.columns = layer.weight.data.shape[1]

        # 初始化一个用于存储每列的缩放因子的向量，大小与权重的列数相同
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        # 初始化样本数量为0
        self.nsamples = 0

        # 存储层的ID和名称，这可能用于区分和跟踪不同的层
        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        # 如果输入是二维的，添加一个维度使其成为三维的
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        # 获取批次的大小
        tmp = inp.shape[0]
        # 如果层是线性层，检查输入的维度，并可能将其重塑
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            # 转置输入，因为PyTorch中的nn.Linear期望批次在第二维
            inp = inp.t()

        # 更新scaler_row向量，考虑到新加入的批次
        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        # 更新样本数量
        self.nsamples += tmp

        # 将输入转为float32类型
        inp = inp.type(torch.float32)
        # 更新scaler_row，根据新的输入调整每一列的缩放因子
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
