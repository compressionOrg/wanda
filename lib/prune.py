import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 

from .ablate import AblateGPT 

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    # 保存模型的原始缓存配置
    use_cache = model.config.use_cache
    # 禁用模型的缓存功能，确保每次输入都会被模型重新处理
    model.config.use_cache = False
    # 获取模型的所有层
    layers = model.model.layers

    # 如果设备映射中有嵌入层使用的设备，更新当前设备变量
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    # 获取模型参数的数据类型
    dtype = next(iter(model.parameters())).dtype
    # 创建一个指定形状和数据类型的零张量，用来存储输入
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    # 设置不需要计算梯度，因为这里只是为了校准模型
    inps.requires_grad = False
    # 初始化一个缓存字典，用于存储处理过程中的信息
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    # 定义一个内部类，用于捕获模型第一层的输入
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            # 存储输入到inps张量中，并更新缓存信息
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            # 抛出异常以中断前向传播
            raise ValueError
    # 将模型的第一层替换为Catcher类实例
    layers[0] = Catcher(layers[0])
    # 遍历数据加载器中的数据批次，并处理
    for batch in dataloader:
        try:
            # 将数据批次送入模型进行处理，由于Catcher的存在会引发异常
            model(batch[0].to(device))
        except ValueError:
            # 捕获异常，但不进行任何操作，目的是为了执行Catcher中的代码
            pass 
    # 恢复模型的第一层为原来的层
    layers[0] = layers[0].module

    # 创建一个与输入张量形状和类型相同的零张量，用于存储输出
    outs = torch.zeros_like(inps)
    # 从缓存中取出attention_mask和position_ids
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    # 恢复模型的缓存设置
    model.config.use_cache = use_cache

    # 返回捕获的输入、初始化的输出、attention_mask和position_ids
    return inps, outs, attention_mask, position_ids


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    # alpha: 一个阈值系数
    # sort_res: 排序后的结果，可能是权重或其他度量的排序数组
    # W_metric: 权重或其他度量的原始矩阵
    # tmp_metric: 用于确定阈值的度量，可能是累积和或其他统计量
    # sum_before: tmp_metric 中元素的累积和

    thres_cumsum = sum_before * alpha 
    # 计算累积和与 alpha 的乘积，得到一个累积的阈值

    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    # 使用广播机制，将 thres_cumsum 调整为与 tmp_metric 相同的形状，并创建一个掩码，
    # 掩码的每个元素表明 tmp_metric 中的对应元素是否小于等于 thres_cumsum

    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    # sort_res[0] 表示取 sort_res 的第一行
    # sort_mask.sum(dim=1, keepdims=True) 在第二维上求和，并保持维度不变，结果是每行的掩码和
    # 减去1来得到索引（因为Python是基于0的索引）
    # torch.gather 根据这些索引从 sort_res[0] 中收集元素，构成一个新的阈值张量 thres

    W_mask = (W_metric <= thres)
    # 创建一个新的掩码 W_mask，标记 W_metric 中所有小于等于阈值 thres 的元素

    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    # 计算稀疏性，即 W_mask 中为 True 的元素数除以 W_mask 中元素总数

    return W_mask, cur_sparsity
    # 返回计算得到的掩码 W_mask 和稀疏性 cur_sparsity


def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:2"), prune_n=0, prune_m=0):
    # 获取模型的层列表
    layers = model.model.layers 

    # 遍历每个层
    for i in range(len(layers)):
        layer = layers[i]
        # 找到需要剪枝的子集层
        subset = find_layers(layer)

        # 遍历子集层中的每个层
        for name in subset:
            # 获取层的权重
            W = subset[name].weight.data 
            # 计算权重的绝对值，作为剪枝的度量
            W_metric = torch.abs(W)
            
            # 判断是否进行结构化剪枝
            if prune_n != 0:
                # 初始化一个与权重形状相同的全零张量，用于创建剪枝掩码
                W_mask = (torch.zeros_like(W) == 1)
                
                # 遍历权重的列
                for ii in range(W_metric.shape[1]):
                    # 判断是否为结构化剪枝的列
                    if ii % prune_m == 0:
                        # 获取当前结构化剪枝列的剪枝度量
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        # 根据剪枝度量，创建剪枝掩码，将对应的权重位置设为零
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                # 计算剪枝阈值，通过将剪枝比例乘以权重总数得到剪枝阈值的位置，并取出该位置的权重值
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel() * args.sparsity_ratio)].cpu()
                # 根据阈值创建剪枝掩码，将小于等于阈值的位置设为True
                W_mask = (W_metric <= thresh)

            # 根据剪枝掩码，将对应的权重位置设为零，完成剪枝操作
            W[W_mask] = 0


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:2"), prune_n=0, prune_m=0):
    # 保存原始模型缓存配置，并暂时禁用它。
    # 确保修剪校准期间不使用之前的计算结果。
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    # 开始加载校准数据，并在加载完成后通知。
    print("loading calibration data")
    dataloader, _ = get_loaders(
        "c4", 
        nsamples=args.nsamples, 
        seed=args.seed, 
        seqlen=model.seqlen, 
        tokenizer=tokenizer
    )
    print("dataset loading complete")

    # 准备校准输入，同时不追踪梯度以提高效率。
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    # 获取模型内部的层列表。
    layers = model.model.layers

    # 遍历每一层进行修剪操作。
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)  # 查找需要修剪的层的子集。

        # 如果模型有设备映射（可能是多GPU情况），则进行相应的设备分配。
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev), 
                outs.to(dev), 
                attention_mask.to(dev), 
                position_ids.to(dev)
            )

        # 初始化一个字典用于存储层的包装器。
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        # 定义添加批处理数据的函数，用于钩子中。
        def add_batch(name):
            # 定义临时函数，获取输入输出并添加到对应的包装层。
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        # 注册前向钩子，并将句柄添加到列表以便之后移除。
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # 对每个校准样本执行前向传播，并收集数据。
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        # 移除之前注册的所有钩子。
        for h in handles:
            h.remove()

        # 对每个子集中的层进行修剪操作。
        for name in subset:
            print(f"pruning layer {i} name {name}")
            # 计算修剪度量，基于权重的绝对值和对应的激活函数
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            # 初始化修剪掩码，开始时全为False。
            W_mask = (torch.zeros_like(W_metric) == 1)

            # 如果设置了结构化修剪参数，则执行结构化修剪。
            if prune_n != 0:
                # 结构化n:m稀疏性
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii+prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                # 非结构化修剪
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            # 最后将掩码为True的权重值设为零，完成修剪。
            subset[name].weight.data[W_mask] = 0

        # 再次对每个样本执行前向传播，可能用于验证修剪效果。
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        # 交换输入和输出的引用，为下一轮或后续操作准备。
        inps, outs = outs, inps

    # 恢复模型的缓存设置。
    model.config.use_cache = use_cache 

    # 清空CUDA缓存，以减少内存消耗。
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    
    # 获取数据加载器和其他相关信息
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)

    # 设置模型缓存的使用信息
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # 设置模型的计算设备
    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    # 创建Catcher类，用于捕获输入数据并存储到inps中
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    
    # 将第一个层替换为Catcher类，以便捕获输入数据
    layers[0] = Catcher(layers[0])
    
    # 遍历数据加载器，获取输入数据并存储到inps中
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    
    # 将第一个层恢复为原来的层
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    # 创建与inps形状相同的张量，用于存储模型的输出
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    # 遍历每个层
    for i in range(len(layers)):
        layer = layers[i]
        # 获取当前层的计算设备信息
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        # 找到需要剪枝的子集层
        subset = find_layers(layer)

        # 创建存储SparseGPT对象的字典
        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        # 创建处理每个子集层的回调函数
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        # 注册回调函数，用于捕获每个子集层的输入和输出数据
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # 处理每个输入样本，获取模型的输出并存储到outs中
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        # 针对每个子集层进行剪枝操作
        for name in gpts:
            print(i, name)
            print('Pruning ...')

            # 使用SparseGPT对象进行剪枝，指定剪枝比例和剪枝数量
            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        # 处理每个输入样本，获取模型的输出并存储到outs中
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        # 更新当前层的权重
        layers[i] = layer 
        torch.cuda.empty_cache()

        # 交换inps和outs的值，用于下一层的计算
        inps, outs = outs, inps

    # 恢复模型的缓存使用设置
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')

    # 获取数据加载器和其他相关信息
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)

    # 设置模型缓存的使用信息
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # 设置模型的计算设备
    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    # 创建Catcher类，用于捕获输入数据并存储到inps中
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    # 将第一个层替换为Catcher类，以便捕获输入数据
    layers[0] = Catcher(layers[0])

    # 遍历数据加载器，获取输入数据并存储到inps中
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass

    # 将第一个层恢复为原来的层
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    # 创建与inps形状相同的张量，用于存储模型的输出
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    # 遍历每个层
    for i in range(len(layers)):
        layer = layers[i]
        # 获取当前层的计算设备信息
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        # 找到需要剪枝的子集层
        subset = find_layers(layer)

        # 创建存储AblateGPT对象的字典
        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        # 创建处理每个子集层的回调函数
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        # 注册回调函数，用于捕获每个子集层的输入和输出数据
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # 处理每个输入样本，获取模型的输出并存储到outs中
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            # 根据剪枝方法选择剪枝掩码
            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            # 使用AblateGPT对象进行剪枝，指定剪枝比例、剪枝掩码和剪枝数量
            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        # 处理每个输入样本，获取模型的输出并存储到outs中
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        # 更新当前层的权重
        layers[i] = layer 
        torch.cuda.empty_cache()

        # 交换inps和outs的值，用于下一层的计算
        inps, outs = outs, inps

    # 恢复模型的缓存使用设置
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
