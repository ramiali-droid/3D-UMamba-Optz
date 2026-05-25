"""
Adapted for S3DIS 8-class training/fine-tuning
High-performance version for powerful GPU
"""
import argparse
import os
from data_utils.S3DISBlockDataLoader_v3 import S3DISDataset  # ← updated
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
from ErrorMatrix import ConfusionMatrix
from torch.nn import DataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
import multiprocessing
from torch.utils.tensorboard import SummaryWriter
from zoneinfo import ZoneInfo

writer = SummaryWriter()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ceiling', 'floor', 'wall', 'column', 'window', 'door', 'furniture', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def parse_args():
    berlin_tz = ZoneInfo("Europe/Berlin")
    berlin_time = datetime.datetime.now(berlin_tz)
    berlin_iso = berlin_time.isoformat().split('T')[0]


    default_log_dir = f'sedis_{berlin_iso}'
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='mamba_msg', help='model name')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training')
    parser.add_argument('--epoch', default=100, type=int, help='Epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=default_log_dir, help='Log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=8192, help='Point Number')
    parser.add_argument('--step_size', type=int, default=5, help='Decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.8, help='Decay rate for lr decay')
    parser.add_argument('--num_category', type=int, default=8, help='num_category')
    parser.add_argument('--weighted_loss', type=bool, default=True, help='weighted loss')
    parser.add_argument('--fold', type=int, default=0, help='Fold 0-2 for cross-validation')
    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('s3dis_8class_cv_rad_0.5')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(args.log_dir or timestr)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    log_string('PARAMETER ...')
    log_string(args)

    root = '/home/ramiali/Downloads/Stanford3dDataset_v1.2_Aligned_Version/merged_rooms/subsampled_0.010/Block_s2_min_final_8192_norm_enhance_rad_0.5'

    NUM_CLASSES = args.num_category
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    # print("start loading training data ...")
    # TRAIN_DATASET = S3DISDataset(split='train', data_root=root, fold=args.fold, npoints=NUM_POINT)
    # print("start loading test data ...")
    # TEST_DATASET = S3DISDataset(split='test', data_root=root, fold=args.fold, npoints=NUM_POINT)

    # trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    #                                               pin_memory=True, drop_last=False)
    # testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    #                                              pin_memory=True, drop_last=False)

    # log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    # log_string("The number of test data is: %d" % len(TEST_DATASET))

    # data_iter = iter(trainDataLoader)
    # points, labels, fps_series, idx_series = next(data_iter)
    # logger.info(f'test dataloader:', points.size(), labels.size(), fps_series.size(), idx_series.size())
    # raise
    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES, fps_n_list=[512, 128, 32]).cuda()
    classifier = DataParallel(classifier)
    criterion = MODEL.get_loss().cuda()
    # weight = torch.tensor(TRAIN_DATASET.labelweights).cuda()

    if args.weighted_loss:
        print("Use weighted loss ...")
        criterion = MODEL.get_loss_weighted().cuda()

    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        #checkpoint = torch.load(str(experiment_dir) + '/checkpoints/pre_model.pth')
        checkpoint = torch.load('/home/ramiali/3dumamba/log/s3dis_8class_cv/sedis_2026-01-29/checkpoints/pre_model.pth')
        start_epoch = 0
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.decay_rate)

    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.learning_rate/100)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECAY = 0.5
    MOMENTUM_DECAY_STEP = args.step_size

    
    # best_iou = 0
    # best_ave_F1_score = 0

    print('Start Training...')
    for fold in range(3):
        print("start loading training data ...")
        TRAIN_DATASET = S3DISDataset(split='train', data_root=root, fold=fold, npoints=NUM_POINT)
        print("start loading test data ...")
        TEST_DATASET = S3DISDataset(split='test', data_root=root, fold=fold, npoints=NUM_POINT)

        trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                                    pin_memory=True, drop_last=False)
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                                                    pin_memory=True, drop_last=False)
        weight = torch.tensor(TRAIN_DATASET.labelweights).cuda()

        log_string("The number of training data is: %d" % len(TRAIN_DATASET))
        log_string("The number of test data is: %d" % len(TEST_DATASET))
        global_epoch = 0
        best_iou = 0
        best_ave_F1_score = 0
        for epoch in range(start_epoch, args.epoch):
            log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))

            # scheduler.step()
            logger.info('Learning rate is: %f' % (optimizer.state_dict()['param_groups'][0]['lr']))

            momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECAY ** (epoch // MOMENTUM_DECAY_STEP))
            if momentum < 0.01:
                momentum = 0.01
            print('BN momentum updated to: %f' % momentum)
            classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

            num_batches = len(trainDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            classifier = classifier.train()


            for i, (points, target, fps_index_array, series_idx_arrays) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
                optimizer.zero_grad()
                points = points.data.numpy()
                points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
                points[:, :, :3] = provider.random_scale_point_cloud(points[:, :, :3])
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                fps_index_array, series_idx_arrays = fps_index_array.long().cuda(), series_idx_arrays.long().cuda()
                points = points.transpose(2, 1)

                pre = classifier(points, fps_index_array, series_idx_arrays)
                pre = pre.contiguous().view(-1, NUM_CLASSES)
                target = target.view(-1)

                loss = criterion(pre, target, weight)
                writer.add_scalar(f"Loss/train - fold {fold}", loss, epoch)
                loss.backward()
                optimizer.step()
                scheduler.step()

                pred_choice = pre.cpu().data.max(1)[1].numpy()
                correct = np.sum(pred_choice == target.cpu().numpy())
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                loss_sum += loss.item()

            log_string('Training mean loss on fold %s: %f' % (str(fold), (loss_sum / num_batches)))
            log_string('Training accuracy on fold %s: %f' % (str(fold), (total_correct / float(total_seen))))

            with torch.no_grad():
                num_batches = len(testDataLoader)
                total_correct = 0
                total_seen = 0
                loss_sum = 0
                total_seen_class = [0 for _ in range(NUM_CLASSES)]
                total_correct_class = [0 for _ in range(NUM_CLASSES)]
                total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

                confusion = ConfusionMatrix(num_classes=args.num_category, labels=classes)

                classifier = classifier.eval()
                log_string('---- EPOCH %03d EVALUATION on FOLD %s----' % (global_epoch + 1, str(fold)))

                for i, (points, target, fps_index_array, series_idx_arrays) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                    points = points.data.numpy()
                    input_points = points
                    points = torch.Tensor(points)
                    points, target = points.float().cuda(), target.long().cuda()
                    fps_index_array, series_idx_arrays = fps_index_array.long().cuda(), series_idx_arrays.long().cuda()
                    points = points.transpose(2, 1)

                    pre = classifier(points, fps_index_array, series_idx_arrays)
                    seg_pred = pre
                    pred_val = seg_pred.contiguous().cpu().data.numpy()
                    pre = pre.contiguous().view(-1, NUM_CLASSES)
                    batch_label = target.cpu().data.numpy()
                    target = target.view(-1, 1)[:, 0]
                    loss = criterion(pre, target, weight)
                    loss_sum += loss

                    pred_val = np.argmax(pred_val, 2)
                    correct = np.sum((pred_val == batch_label))
                    total_correct += correct
                    total_seen += (BATCH_SIZE * NUM_POINT)

                    pred_val_flatten = pred_val.reshape(-1,1)
                    batch_label_flatten = batch_label.reshape(-1,1)
                    confusion.update(pred_val_flatten, batch_label_flatten)

                    for l in range(NUM_CLASSES):
                        total_seen_class[l] += np.sum((batch_label == l))
                        total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                        total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

                ave_F1_score, miou, acc = confusion.summary()
                mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6))

                writer.add_scalar(f"Loss/Eval - fold {fold}", loss_sum / float(num_batches), epoch)
                log_string('eval mean loss on fold %s: %f' % (str(fold), loss_sum / float(num_batches)))
                log_string('eval point avg class IoU on fold %s: %f' % (str(fold), mIoU))
                log_string('eval point avg class IoU-2 on fold %s: %f' % (str(fold), miou))
                log_string('eval point accuracy on fold %s: %f' % (str(fold), total_correct / float(total_seen)))
                log_string('eval ave_F1_score on fold %s: %f' % (str(fold), ave_F1_score))
                writer.flush()
                if mIoU >= best_iou:
                    best_iou = mIoU
                    logger.info('Save model...')
                    # savepath = str(checkpoints_dir) + str(fold) + '/best_model.pth'
                    fold_dir = checkpoints_dir.joinpath(f'{str(fold)}/')
                    fold_dir.mkdir(exist_ok=True)
                    savepath = str(fold_dir) + '/best_model.pth'
                    log_string('Saving at %s' % savepath)
                    state = {
                        'epoch': epoch,
                        'fold': str(fold),
                        'class_avg_iou': mIoU,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ave_F1_score': ave_F1_score,
                    }
                    torch.save(state, savepath)

            global_epoch += 1
    writer.close()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    args = parse_args()
    main(args)