import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from PIL import Image, ImageDraw
from tqdm import tqdm
import torch.distributed as dist
import shutil
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch import nn
from scipy.spatial import cKDTree

from dataset import RNGDet_dataset
from models.detr import build_model
from main_val import valid


def create_directory(dir, delete=False):
    if os.path.isdir(dir) and delete:
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)


def train(args):
    # ==============
    global writer
    if args.multi_GPU:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(f'cuda:{args.local_rank}')
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        train_loader, train_sampler = RNGDet_dataset(args)
        RNGDetNet, criterion = build_model(args)
        RNGDetNet.cuda()
        RNGDetNet = torch.nn.parallel.DistributedDataParallel(RNGDetNet, device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              find_unused_parameters=True)
        model_without_ddp = RNGDetNet.module
    else:
        train_loader = RNGDet_dataset(args)
        RNGDetNet, criterion = build_model(args)
        RNGDetNet.cuda()
        model_without_ddp = RNGDetNet

    # 如果args.current_best_model为真，则加载模型的权重，以便从上次训练的地方继续训练。
    if args.current_best_model:
        best_model = torch.load(args.current_best_model, map_location='cpu')
        if 'model' in best_model:
            best_model = best_model['model']
        state_dict = model_without_ddp.state_dict()
        state_dict.update(best_model)
        #      model_without_ddp.load_state_dict(best_model, strict=(not args.no_strict_load))
        model_without_ddp.load_state_dict(state_dict, strict=False)

    # 分组pytorch模型的参数，以供优化。列表中的每个字典代表一组将一起优化的参数。backbone的学习率是lr_backbone，一般较低，其他的是lr，一般较高。
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    # AdamW优化器,可以有效处理权重衰减(weight)问题
    opt = torch.optim.AdamW(param_dicts, lr=args.lr,
                            weight_decay=args.weight_decay)
    # 学习率调度器,每隔lr_drop个epoch，学习率乘以gamma
    sched = MultiStepLR(opt, [20, 30, 40], 0.1)
    # 如果args.resume为真，则加载模型的权重和优化器的状态，以便从上次训练的地方继续训练。
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            opt.load_state_dict(checkpoint['optimizer'])
            sched.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        sched.step(args.start_epoch - 1)
    # 代码检查当前进程是否为主进程（主要用于分布式训练设置）
    if args.local_rank == 0:
        # ============== 
        if args.multi_scale:
            args.savedir = f'{args.savedir}_multi'
        if args.instance_seg:
            args.savedir = f'{args.savedir}_ins'
        create_directory(f'./{args.savedir}/tensorboard', delete=True)
        create_directory(f'./{args.savedir}/tensorboard_past')
        create_directory(f'./{args.savedir}/train', delete=True)
        create_directory(f'./{args.savedir}/valid', delete=True)
        create_directory(f'./{args.savedir}/checkpoints')
        # 从torch.utils.tensorboard导入SummaryWriter，用于记录训练过程
        writer = SummaryWriter(args.savedir + '/tensorboard')

    # 从pytorch的nn模块中导入Sigmoid函数，常用于二分类问题的输出层，将输出值映射到0-1之间，表示概率
    sigmoid = nn.Sigmoid()

    # =====================================
    # 将模型设置为训练模式，这样在模型中的dropout和batch normalization层会起作用，否则会被冻结（这些层在训练模式和评估模式的行为不同）
    RNGDetNet.train()
    # 这两个变量用于在训练过程中记录最好的f1值和最近10个epoch中最好的f1值
    best_f1 = 0
    best_f1_last_10_epoch = 0
    # 这行代码是训练循环的开始。模型将被训练args.nepochs轮次。一个轮次是对整个训练数据集的一次完整遍历。在每个轮次中，模型的权重都会被更新以最小化损失函数。
    for epoch in range(args.nepochs):
        if args.multi_GPU:
            # 分布式训练中调用set_epoch方法，以确保每个GPU看到的数据子集是唯一且不同的，这样可以避免重复训练
            train_sampler.set_epoch(epoch)
        # tqdm是一个Python库，用于在循环中显示进度条。在这里，tqdm用于显示训练过程中的进度条。unit='img'表示进度条的单位是图像。total=len(train_loader)表示进度条的总长度是训练数据集的长度。
        with tqdm(total=len(train_loader), unit='img') as pbar:
            # 循环遍历train_loader. train_loader是一个数据加载器，每次迭代返回一个批次的数据。enumerate函数用于获取每次迭代的索引i和值data
            # 这允许模型一次在一个批次的数据上训练，而不是一次在整个数据集上训练。
            for i, data in enumerate(train_loader):
                # sat卫星图像, historical_map历史地图, label_masks标签掩码, gt_prob地面真实概率, gt_coord地面真实坐标, gt_mask地面真实掩码, list_len批次数据点列表的长度。
                sat, historical_map, label_masks, gt_prob, gt_coord, gt_mask, list_len = data
                # 将数据转换为张量(LongTensor或FloatTensor)，利用.cuda()方法将其移动到GPU上
                sat, historical_map, label_masks, gt_prob, gt_coord, gt_mask = \
                    sat.type(torch.FloatTensor).cuda(), \
                        historical_map.type(torch.FloatTensor).cuda(), \
                        label_masks.type(torch.FloatTensor).cuda(), \
                        gt_prob.type(torch.LongTensor).cuda(), \
                        gt_coord.type(torch.FloatTensor).cuda(), \
                        gt_mask.type(torch.FloatTensor).cuda()
                # 模型RngetNet的前向传播，应用于sat和historical_map数据，返回模型的输出，输出存储在outputs变量中。
                outputs = RNGDetNet(sat, historical_map)
                # 遍历label_masks张量的第一维度的范围，生成一个字典列表target，包含一个批次中每个数据点的GT概率gt_prob、标签掩码label_masks、GT坐标gt_coord、GT掩码gt_mask。
                # 这个字典列表是用于计算损失函数的GT target
                targets = [
                    {'labels': gt_prob[x, :list_len[x]], 'masks': label_masks[x], 'boxes': gt_coord[x, :list_len[x]],
                     'instance_masks': gt_mask[x, :list_len[x]]} for x in range(label_masks.shape[0])]
                # 计算损失函数critetion，应用于模型的输出outputs和目标targets，返回损失值loss。其结果是一个字典，包含损失函数的各个部分，可以用来更新模型的权重。
                loss_dict = criterion(outputs, targets)
                # args.instance_seg: 判断是否进行实例分割，如果是，则损失函数包括交叉熵损失、坐标损失、分割损失、实例分割损失，否则只包括交叉熵损失、坐标损失、分割损失。（计算损失的方式有所不同）
                if args.instance_seg:
                    # 总损失为交叉熵损失loss_ce、坐标损失loss_coord、分割损失loss_seg、实例分割损失loss_instance_seg的和。
                    loss_ce, loss_coord, loss_seg, loss_instance_seg = loss_dict['loss_ce'], loss_dict['loss_bbox'] * 5, \
                    loss_dict['loss_seg'], loss_dict['loss_instance_seg']
                    loss = loss_ce + loss_coord + loss_seg + loss_instance_seg
                else:
                    # 总损失为交叉熵损失loss_ce、坐标损失loss_coord、分割损失loss_seg的和。（没有实例分割损失）
                    loss_ce, loss_coord, loss_seg = loss_dict['loss_ce'], loss_dict['loss_bbox'] * 5, loss_dict[
                        'loss_seg']
                    loss = loss_ce + loss_coord + loss_seg
                # 从模型的输出outputs中获取预测的坐标pred_coords和预测的概率pred_probs
                pred_coords = outputs['pred_boxes'][-1]
                pred_probs = outputs['pred_logits'][-1]
                # 用于更新模型权重。首先将优化器opt的梯度归零（必须清零，因为pytorch会积累梯度）
                # 然后计算损失相对于模型参数的梯度（ loss.backward() ）
                # 最后根据当前的梯度更新模型的权重（ opt.step() ）
                # 这三个步骤是训练模型的核心步骤，对每个批次的数据都会执行这三个步骤。
                opt.zero_grad()
                loss.backward()
                opt.step()

                # ====================== vis
                # 以下代码用于可视化训练过程中的损失值和预测结果

                # 条件语句，用于检查当前进程是否为主进程（主要用于分布式训练设置）
                # 这个条块内的代码只有在主进程中才会执行，确保某些任务只在主进程中执行，比如记录训练过程中的损失值和预测结果，保存模型的权重等，避免算力浪费。
                if args.local_rank == 0:
                    # 用tensorboard的SummaryWriter对象的add_scalar方法（ writer.add_scalar() ），记录训练过程中的损失值。
                    # 记录的损失值包括交叉熵损失loss_ce、坐标损失loss_coord、分割损失loss_seg。
                    # 这些损失在每次迭代中都会被记录，可以在tensorboard中查看，x轴的值是迭代次数，为i+epoch*len(train_loader)。
                    writer.add_scalar('train/loss_ce', loss_ce, i + epoch * len(train_loader))
                    writer.add_scalar('train/loss_coord', loss_coord, i + epoch * len(train_loader))
                    writer.add_scalar('train/loss_seg', loss_seg, i + epoch * len(train_loader))

                    # 如果args.instance_seg为真，则记录实例分割损失loss_instance_seg。
                    if args.instance_seg:
                        writer.add_scalar('train/loss_instance_seg', loss_instance_seg, i + epoch * len(train_loader))

                    # 每100次迭代，可视化记录一次预测结果。
                    # 从模型的输出中提取预测的二值掩码pred_binary（即二元分割的预测）和预测的关键点pred_keypoints。
                    # 并转换为图像格式（具体方法：用sigmoid函数将值限制在0和1之间，然后缩放到0-255的范围进行可视化）
                    if i % 100 == 0:
                        # vis
                        # outputs['pred_masks'][-1,0]是预测的二值分割，outputs['pred_masks'][-1,1]是预测的关键点。这些预测可能以logits的形式给出，需要通过sigmoid函数转换为概率。
                        # -1表示从此张量中选择最后一批预测，0表示第一个通道（存储了二值分割），1表示第二个通道（存储了关键点）。
                        pred_binary = sigmoid(outputs['pred_masks'][-1, 0]) * 255
                        pred_keypoints = sigmoid(outputs['pred_masks'][-1, 1]) * 255

                        # vis
                        # 创建一个图像，包括原始卫星图像、历史地图、GT二值分割、GT关键点、预测二值分割、预测关键点。

                        # 创建一个新的图像dst，大小为args.ROI_SIZE*4+5 x args.ROI_SIZE*2+5，用于存储可视化结果。
                        dst = Image.new('RGB', (args.ROI_SIZE * 4 + 5, args.ROI_SIZE * 2 + 5))
                        # 从不同的张量中提取数据，转换为numpy数组，然后转换为图像格式，存储在dst中。
                        # sat是卫星图像，historical_map是历史地图，label_masks是GT二值分割，gt_keypoint是GT关键点，pred_binary是预测二值分割，pred_keypoint是预测关键点。
                        # 每个tensor通过cpu()移动到CPU，detach()从计算图中分离，numpy()将数据转换为numpy数组，乘以255转换为0-255的范围，astype(np.uint8)转换为8位无符号整数格式。
                        # 最后通过Image.fromarray方法将numpy数组转换为PIL图像。
                        sat = Image.fromarray((sat[-1].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8))
                        history = Image.fromarray((historical_map[-1, 0].cpu().detach().numpy() * 255).astype(np.uint8))
                        gt_binary = Image.fromarray((label_masks[-1, 0].cpu().detach().numpy() * 255).astype(np.uint8))
                        gt_keypoint = Image.fromarray(
                            (label_masks[-1, 1].cpu().detach().numpy() * 255).astype(np.uint8))
                        pred_binary = Image.fromarray((pred_binary.cpu().detach().numpy()).astype(np.uint8))
                        pred_keypoint = Image.fromarray((pred_keypoints.cpu().detach().numpy()).astype(np.uint8))

                        # 将这些图像粘贴到dst图像中，产生一个大图像，包括原始卫星图像、历史地图、GT二值分割、GT关键点、预测二值分割、预测关键点。
                        dst.paste(sat, (0, 0))
                        dst.paste(history, (0, args.ROI_SIZE))
                        dst.paste(gt_binary, (args.ROI_SIZE, 0))
                        dst.paste(gt_keypoint, (args.ROI_SIZE * 2, 0))
                        dst.paste(pred_binary, (args.ROI_SIZE, args.ROI_SIZE))
                        dst.paste(pred_keypoint, (args.ROI_SIZE * 2, args.ROI_SIZE))

                        # 如果args.instance_seg为真，则记录GT实例分割和预测实例分割。
                        if args.instance_seg:

                            # 从outputs中提取GT的实例分割，转换为图像格式，存储在dst中。
                            # gt_instance_mask是GT的实例分割，gt_mask[-1]是一个张量，代表最后一个批次的数据的GT掩码。（这里仅处理最后一个批次仅为了进行某种形式的可视化和评估）
                            # torch.sum函数用于沿着零维度对实例分割求和，有效地创建一个单一的图像，其中每个像素的强度对应于该像素处的实例数量
                            # np.clip函数用于将图像的强度限制在0-255的范围内，astype(np.uint8)将图像转换为8位无符号整数格式
                            # Image.fromarray方法将numpy数组转换为PIL图像，paste方法将GT实例分割粘贴到dst图像中。
                            gt_instance_mask = Image.fromarray(
                                np.clip((torch.sum(gt_mask[-1], dim=0) * 255).cpu().detach().numpy(), 0, 255).astype(
                                    np.uint8))
                            dst.paste(gt_instance_mask, (args.ROI_SIZE * 3, 0))

                            # 从outputs中提取预测的实例分割，转换为图像格式，存储在dst中。
                            # enumerate()函数用于将可迭代对象转换成枚举对象，在这里用于获取索引ii和值x。
                            # 其中x是outputs['pred_instance_masks'][-1]的元素，outputs['pred_instance_masks'][-1]是一个张量，代表最后一个批次的数据的预测实例分割。
                            # args.logit_threshold是一个阈值，用于过滤预测的实例分割，只保留概率大于阈值的实例。通过这种方式创建二值分割。
                            # torch.sum函数用于沿着零维度对实例分割求和，有效地创建一个单一的图像，其中每个像素的强度对应于该像素处的实例数量
                            # 然后通过dst.paste方法将预测的实例分割粘贴到dst图像中，位置在(args.ROI_SIZE*3,args.ROI_SIZE)。
                            # PS：在深度学习中，一个张量的维度通常是(batch_size,channels,height,width)，这里的实例分割是一个(batch_size,1,height,width)的张量。
                            # PS：batch_size是批次大小，channels是多通道图像的通道数(普遍意义是特征数量)，height是高度，width是宽度（宽度和高度合称空间维度）。
                            # PS：所以，沿着第几个维度求和，本质是按某个维度进行压缩，该维度会被压缩成1。具体表现为将这个维度的所有元素相加，得到一个新的张量，这个张量的维度比原来的张量少一个。
                            pred_logits = pred_probs.softmax(dim=1)
                            pred_logits = [x.unsqueeze(0) for ii, x in
                                           enumerate(outputs['pred_instance_masks'][-1].sigmoid()) if
                                           pred_logits[ii][0] >= args.logit_threshold]
                            if len(pred_logits):
                                pred_instance_mask = torch.cat(pred_logits, dim=0)
                                pred_instance_mask = Image.fromarray(
                                    np.clip((torch.sum(pred_instance_mask, dim=0) * 255).cpu().detach().numpy(), 0,
                                            255).astype(np.uint8))
                                dst.paste(pred_instance_mask, (args.ROI_SIZE * 3, args.ROI_SIZE))

                                # 在dst图像中绘制一系列橙色的圆圈，用于标记预测的关键点的位置。
                        # 通过ImageDraw.Draw方法创建一个可绘制的对象draw，然后通过draw.ellipse方法绘制橙色的圆圈。
                        draw = ImageDraw.Draw(dst)
                        # 双重循环，遍历3x2的矩阵，绘制橙色的圆圈。对于网格中的每个位置的中心(delta_x,delta_y)，绘制一个橙色的圆圈，除了位置为(2,1)的单元格。
                        for ii in range(3):
                            for jj in range(2):
                                if not (ii == 2 and jj == 1):
                                    delta_x = ii * args.ROI_SIZE + args.ROI_SIZE // 2
                                    delta_y = jj * args.ROI_SIZE + args.ROI_SIZE // 2
                                    draw.ellipse([delta_x - 1, delta_y - 1, delta_x + 1, delta_y + 1], fill='orange')
                                    # 如果对于最后一批数据有可用的GT keypoint(list_len[-1])不为零，则在这些关键点的位置绘制青色的圆圈。
                                    if list_len[-1]:
                                        for kk in range(list_len[-1]):
                                            v_next = gt_coord.cpu().detach().numpy()[-1, kk]
                                            draw.ellipse([delta_x - 1 + (v_next[0] * args.ROI_SIZE // 2),
                                                          delta_y - 1 + (v_next[1] * args.ROI_SIZE // 2), \
                                                          delta_x + 1 + (v_next[0] * args.ROI_SIZE // 2),
                                                          delta_y + 1 + (v_next[1] * args.ROI_SIZE // 2)], fill='cyan')

                                    for jj in range(pred_coords.shape[0]):
                                        # 最后在预测关键点的位置绘制黄色或粉色的圆圈。如果预测的概率小于0.5，则绘制黄色的圆圈，否则绘制粉色的圆圈。
                                        v = pred_coords[jj]
                                        v = [delta_x + (v[0] * args.ROI_SIZE // 2),
                                             delta_y + (v[1] * args.ROI_SIZE // 2)]
                                        if pred_probs[jj][0] < pred_probs[jj][1]:
                                            draw.ellipse((v[0] - 1, v[1] - 1, v[0] + 1, v[1] + 1), fill='yellow',
                                                         outline='yellow')
                                        else:
                                            draw.ellipse((v[0] - 1, v[1] - 1, v[0] + 1, v[1] + 1), fill='pink',
                                                         outline='pink')
                        # 通过dst.convert('RGB').save方法将dst图像保存为png格式的文件，文件名为i.png，存储在./{args.savedir}/train/目录下。
                        dst.convert('RGB').save(f'./{args.savedir}/train/{i}.png')

                # 这段代码的作用：更新进度条，显示当前的epoch和损失值。
                # 如果在分布式训练中，调用dist.barrier()方法，确保所有进程都到达了代码的这个位置，然后再更新进度条。防止进度条的更新混乱。
                if args.multi_GPU:
                    dist.barrier()

                # 代码检查是否启动实例分割(args.instance_seg)，如果是，则在进度条中显示当前epoch号和四个不同损失组建的当前值：交叉熵损失loss_ce、分割损失loss_seg、实例分割损失loss_instance_seg、坐标损失loss_coord。
                if args.instance_seg:
                    pbar.set_description(
                        f'===Epoch: {epoch} | ce/seg/instance/coord: {round(loss_ce.item(), 3)}/{round(loss_seg.item(), 3)}/{round(loss_instance_seg.item(), 3)}/{round(loss_coord.item(), 3)} ')
                # 如果没有启动实例分割，则在进度条中显示当前epoch号和三个不同损失组建的当前值：交叉熵损失loss_ce、分割损失loss_seg、坐标损失loss_coord。
                else:
                    pbar.set_description(
                        f'===Epoch: {epoch} | ce/seg/coord: {round(loss_ce.item(), 3)}/{round(loss_seg.item(), 3)}/{round(loss_coord.item(), 3)} ')
                # 更新进度条
                pbar.update()
                # break

        # 条件语句，用于检查当前进程是否为主进程（主要用于分布式训练设置）
        if args.local_rank == 0:
            # 保存模型的权重，以便从上次训练的地方继续训练。
            torch.save({'model': model_without_ddp.state_dict(),
                                'optimizer': opt.state_dict(),
                                'lr_scheduler': sched.state_dict(),
                                'epoch': epoch,
                                'args': args,
                                },
                       os.path.join(args.savedir + '/checkpoints', f"RNGDetNet_{epoch}.pt"))
            print(f'Save checkpoint {epoch}')
            print('Start evaluation.....')
            precision, recall, f1 = evaluate(args, model_without_ddp)
            if f1 > best_f1:
                best_f1 = f1
                torch.save({'model': model_without_ddp.state_dict(),
                                'optimizer': opt.state_dict(),
                                'lr_scheduler': sched.state_dict(),
                                'epoch': epoch,
                                'args': args,
                                },
                           os.path.join(args.savedir + '/checkpoints', f"RNGDetNet_best.pt"))
            if epoch > args.nepochs - 10 and f1 > best_f1_last_10_epoch:
                best_f1_last_10_epoch = f1
                torch.save({'model': model_without_ddp.state_dict(),
                                'optimizer': opt.state_dict(),
                                'lr_scheduler': sched.state_dict(),
                                'epoch': epoch,
                                'args': args,
                                },
                           os.path.join(args.savedir + '/checkpoints', f"RNGDetNet_best_last_10_epoch.pt"))
            print(f'precision/recall/f1: {precision}/{recall}/{f1}')
            writer.add_scalar('eval/precision', precision, epoch)
            writer.add_scalar('eval/recall', recall, epoch)
            writer.add_scalar('eval/f1', f1, epoch)
        if args.multi_GPU:
            dist.barrier()
        RNGDetNet.train()
        sched.step()

# 评估函数，用于评估模型的性能。
def evaluate(args, RNGDetNet):
    # 计算评估指标的函数，用于计算预测的二值分割和GT二值分割之间的准确率、召回率和F1值。
    # 输入参数为预测的点pred_points和GT的点gt_points，用cKDTree函数构建一个KD树，然后计算预测点到GT点的距离。
    # 通过设置一个阈值thr，计算预测点到GT点的距离小于阈值thr的点的数量，除以预测点的总数，得到召回率。
    def calculate_scores(gt_points, pred_points):
        gt_tree = cKDTree(gt_points)
        if len(pred_points):
            pred_tree = cKDTree(pred_points)
        else:
            return 0, 0, 0
        thr = 3
        # 用ckdtree的query方法计算预测点到GT点的距离，k=1表示只返回最近的一个点。
        # dis_gt2pred是一个列表，包含了每个预测点到GT点的最近距离。dis_pred2gt是一个列表，包含了每个GT点到预测点的最近距离。
        dis_gt2pred, _ = pred_tree.query(gt_points, k=1)
        dis_pred2gt, _ = gt_tree.query(pred_points, k=1)
        # 计算召回率: 即距离小于阈值（thr）的地面真实点的数量占总地面真实点的比例。
        # 计算准确率: 即距离小于阈值的预测点的数量占总预测点的比例。
        recall = len([x for x in dis_gt2pred if x < thr]) / len(dis_gt2pred)
        acc = len([x for x in dis_pred2gt if x < thr]) / len(dis_pred2gt)
        # 计算F1值: F1 = 2 * recall * acc / (recall + acc), F1值是准确率和召回率的调和平均数，（在1（完美的精确度和召回率）和0（最差）之间变化）
        r_f = 0
        if acc * recall:
            r_f = 2 * recall * acc / (acc + recall)
        return acc, recall, r_f

    # 计算像素级评估指标的函数，用于计算预测的二值分割pred_mask和GT二值分割gt_mask之间的准确率、召回率和F1值。
    def pixel_eval_metric(pred_mask, gt_mask):
        # tuple2list函数，用于将元组转换为列表。元组是一个有序的不可变的序列，列表是一个有序的可变的序列。
        def tuple2list(t):
            return [[t[0][x], t[1][x]] for x in range(len(t[0]))]

        # 通过np.where函数，获取GT二值分割和预测二值分割中非零元素的坐标，存储在gt_points和pred_points中。
        gt_points = tuple2list(np.where(gt_mask != 0))
        pred_points = tuple2list(np.where(pred_mask != 0))

        # 调用calculate_scores函数，计算预测的点pred_points和GT的点gt_points之间的准确率、召回率和F1值。
        return calculate_scores(gt_points, pred_points)

    # 将模型设置为评估模式，这样在模型中的dropout和batch normalization层会被冻结（这些层在训练模式和评估模式的行为不同）
    RNGDetNet.eval()
    # 调用定义在main_val.py中的函数valid，用于验证参数的有效性。
    valid(args, RNGDetNet)

    # 初始化一个空列表scores，用于存储每个数据点的评估指标（准确率、召回率、F1值）。
    scores = []
    # 遍历./{args.savedir}/valid/skeleton目录下的所有文件，获取文件名，然后打开文件，转换为numpy数组，存储在pred_graph中。
    for name in os.listdir(f'./{args.savedir}/valid/skeleton'):
        pred_graph = np.array(Image.open(f'./{args.savedir}/valid/skeleton/{name}'))[args.ROI_SIZE:-args.ROI_SIZE,
                     args.ROI_SIZE:-args.ROI_SIZE]
        gt_graph = np.array(Image.open(f'./dataset/segment/{name}'))
        scores.append(pixel_eval_metric(pred_graph, gt_graph))
    # 返回评估指标的平均值，round函数用于四舍五入，保留三位小数。
    return round(sum([x[0] for x in scores]) / (len(scores) + 1e-7), 3), \
        round(sum([x[1] for x in scores]) / (len(scores) + 1e-7), 3), \
        round(sum([x[2] for x in scores]) / (len(scores) + 1e-7), 3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HDMapNet training.')
    # logging config
    parser.add_argument("--savedir", type=str)

    # nuScenes config
    parser.add_argument('--dataroot', type=str)

    # loss config
    # 损失函数各个组成部分的缩放因子。
    # scale_seg是分割损失的缩放因子，scale_var是方差损失的缩放因子，scale_dist是距离损失的缩放因子，scale_direction是方向损失的缩放因子。
    parser.add_argument("--scale_seg", type=float, default=1.0)
    parser.add_argument("--scale_var", type=float, default=1.0)
    parser.add_argument("--scale_dist", type=float, default=1.0)
    parser.add_argument("--scale_direction", type=float, default=0.2)

    # test 用于启动测试模式，如果设置为真，则只进行测试，不进行训练。
    parser.add_argument('--test', default=False, action='store_true')

    # 优化过程的参数
    # lr是学习率，lr_backbone是骨干网络的学习率，batch_size是批次大小，weight_decay是权重衰减，epochs是训练的轮数，lr_drop是学习率下降的轮数。
    # 权重衰减与L2正则化有关，是一种防止过拟合的技术，通过在损失函数中增加一个权重衰减项/L2正则化项，使得权重的值变小，从而减小模型的复杂度。
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    # clip_max_norm是梯度裁剪的最大范数，每次梯度更新不能超过一个指定值，用于防止梯度爆炸。
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    # frozen_weights是一个路径参数，用于调用一个现有的权重文件，如果设置了这个参数，只有mask head会被训练。
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    # backbone是一个字符串参数，用于指定使用的卷积骨干网络的名称。
    parser.add_argument('--backbone', default='resnet18', type=str,
                        help="Name of the convolutional backbone to use")
    # dilation是一个布尔参数，如果设置为真，则在最后一个卷积块（DC5）中用扩张卷积代替步长卷积。——可以提高感受野，减少参数量，能涨点，是分割任务中常用的技术。
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    # Position embedding用于指定在图像特征之上使用的位置嵌入的类型。位置嵌入是一种用于处理序列数据的技术，用于为序列中的每个元素提供位置信息。
    # sine: 使用正弦和余弦函数生成位置嵌入，learned: 使用学习的位置嵌入。
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # 调用一个现有的权重文件，路径参数为--current_best_model，具体设置在run_train_*.sh文件中。
    parser.add_argument('--current_best_model', default='', type=str,
                        help="Checkpoint file to load the model from")
    # 除了加载模型权重外，还可以加载优化器和学习率调度器的状态。
    parser.add_argument('--resume', default='', type=str,
                        help="restart")

    # * Transformer
    # 这个参数是一个整数，用于指定编码器中每一层的通道数，即每层处理的特征数量。
    parser.add_argument('--num_channels', default=128 + 64, type=int,
                        help="Number of encoding layers in the transformer")
    #这个参数是一个整数，用于指定编码器中的层数。默认6说明编码部分将堆叠6个相同的层
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    # 这个参数是一个整数，用于指定解码器中的层数。默认6说明解码部分将堆叠6个相同的层
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    # dim_feedforward是一个整数参数，用于指定transformer块中的前馈层的中间大小。
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    # hidden_dim是一个整数参数，用于指定transformer中的嵌入大小（注意力的维度），即每个输入标记都会被转换为一个hidden_dim维的向量（q,k,v）。
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    # dropout是一个浮点数参数，用于指定transformer中的dropout概率。
    parser.add_argument('--dropout', default=0, type=float,
                        help="Dropout applied in the transformer")
    # nheads是一个整数参数，用于指定transformer中的注意力头的数量。
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    # num_queries是一个整数参数，用于指定transformer中的查询槽的数量。默认10说明模型将预测10个查询槽，在本文中，查询槽是预测的关键点的位置。
    parser.add_argument('--num_queries', default=10, type=int,
                        help="Number of query slots")
    # aux_loss是一个布尔参数，如果设置为真，则在每个解码器层中计算一个额外的损失( 预先归一化 )，用于监督模型的训练。
    # 预归一化是指在执行实际计算之前的先对输入进行归一化（如层归一化）
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    # masks是一个布尔参数，如果设置为真，则训练分割头。分割头是一个额外的卷积层，用于预测图像中每个像素的类别。
    parser.add_argument('--masks', action='store_true', default=False,
                        help="Train segmentation head if the flag is provided")

    # Loss
    # no_aux_loss是一个布尔参数，如果设置为真，则禁用辅助解码损失（每层的损失）。
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    # cost_class是一个浮点数参数，用于指定匹配成本中的类别系数。
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    # cost_bbox是一个浮点数参数，用于指定匹配成本中的L1框系数。L1框系数是指在匹配成本中的边界框损失的权重，用于计算预测边界框和GT边界框之间的差异。L1框系数越大，边界框损失的权重越大。
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    # --set_cost_giou参数用于设置匹配成本中的广义交并比（GIoU）框系数。匹配成本是在训练模型时，用于匹配预测对象和真实对象的度量。
    # GIoU框系数影响了GIoU计算在匹配过程中的权重。GIoU是用于衡量两个边界框重叠程度的度量。
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients 损失系数
    # mask_loss_coef是一个浮点数参数，用于指定分割损失的系数。分割损失是指模型预测的分割掩码和GT分割掩码之间的差异。
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    # dice_loss_coef是一个浮点数参数，用于指定Dice损失的系数。Dice损失是一种用于评估两个样本之间相似性的损失函数，用于评估分割模型的性能。
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    # bbox_loss_coef是一个浮点数参数，用于指定边界框损失的系数。边界框损失是指模型预测的边界框和GT边界框之间的差异。
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    # giou_loss_coef是一个浮点数参数，用于指定GIoU损失的系数。GIoU损失是一种用于评估两个边界框之间重叠程度的损失函数，用于评估边界框的性能。
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    # local_rank是一个整数参数，用于指定当前进程的本地排名。本地排名是指在分布式训练中，每个进程在本地的排名。
    parser.add_argument('--local_rank', default=0, type=int)
    # device是一个字符串参数，用于指定当前设备的名称。默认为cuda:0，表示使用编号为0的GPU进行训练。
    parser.add_argument('--device', default='cuda:0', type=str)
    # eos_coef是一个浮点数参数，用于指定无对象类的相对分类权重。无对象类是指在目标检测任务中，没有目标的类别
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # multi_GPU是一个布尔参数，如果设置为真，则启动多GPU训练。
    parser.add_argument('--multi_GPU', action='store_true')
    # nepochs是一个整数参数，用于指定训练的轮数。
    parser.add_argument("--nepochs", type=int, default=1000)
    # nworkers是一个整数参数，用于指定数据加载器的工作进程数。
    parser.add_argument("--nworkers", type=int, default=4)
    # ROI_SIZE是一个整数参数，用于指定感兴趣区域的大小。
    parser.add_argument("--ROI_SIZE", type=int, default=256)
    # orientation_channels是一个整数参数，用于指定方向通道的数量。
    parser.add_argument("--orientation_channels", type=int, default=2)
    # segmentation_channels是一个整数参数，用于指定分割通道的数量。
    parser.add_argument("--segmentation_channels", type=int, default=3)
    # noise是一个整数参数，用于指定噪声的大小。
    parser.add_argument("--noise", type=int, default=7)

    # image_size是一个整数参数，用于指定图像的大小。
    parser.add_argument("--image_size", type=int, default=4096)
    # logit_threshold是一个浮点数参数，用于指定预测的阈值。
    parser.add_argument("--logit_threshold", type=float, default=0.8)
    # candidate_filter_threshold是一个整数参数，用于指定候选过滤器的阈值。
    parser.add_argument("--candidate_filter_threshold", type=int, default=50)
    # extract_candidate_threshold是一个浮点数参数，用于指定提取候选的阈值。
    parser.add_argument("--extract_candidate_threshold", type=float, default=0.65)
    # alignment_distance是一个整数参数，用于指定对齐距离。
    parser.add_argument("--alignment_distance", type=int, default=10)
    parser.add_argument("--filter_distance", type=int, default=10)
    # 是否在模型中使用多尺度处理
    parser.add_argument("--multi_scale", action='store_true')
    # 是否在模型中执行实例分割
    parser.add_argument("--instance_seg", action='store_true')
    # 是否在模型中执行边界处理
    parser.add_argument("--process_boundary", action='store_true')

    # 通过parse_args()方法解析参数，返回一个命名空间，其中包含了所有的参数。
    args = parser.parse_args()
    # 如果没有设置保存目录，则设置默认的保存目录为./{args.savedir}，即当前目录下的{args.savedir}目录。
    train(args)
