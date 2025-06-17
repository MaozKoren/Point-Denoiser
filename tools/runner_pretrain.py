import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from datasets.noise import Noise
# from pointnet2_ops import pointnet2_utils
from pointnet import pointnet2 as pointnet2_utils
from tools.helper_functions import save_training_img, perturb_points, add_noise_in_sphere, plot_training_curve_loss, plot_training_curve_acc,  plot_neighborhood_lables, plot_filtered_points, percentage_noise_inside_shape
from models.Point_MAE import Point_Denoiser

train_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudRotatePerturbation(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudJitter(),
        # data_transforms.PointcloudRandomInputDropout(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]


import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def train_denoiser(config, args, encoder, train_dataloader, start_epoch, logger, checkpoint_dir='checkpoints'):
    # Create directory to save checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize the Point_Denoiser model with the encoder
    model = Point_Denoiser(config.model, encoder).cuda()

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print(f'Max Epochs: {config.max_epoch}')

    # AverageMeter for timing and performance tracking
    batch_time = AverageMeter()
    data_time = AverageMeter()

    for epoch in range(start_epoch, config.max_epoch + 1):
        model.train()
        total_train_loss = 0.0
        epoch_start_time = time.time()

        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            # Measure data loading time
            batch_start_time = time.time()
            data_time.update(time.time() - batch_start_time)

            # Move data to GPU and apply transformations
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                points = misc.fps(points, npoints)
            else:
                raise NotImplementedError(f'Dataset {dataset_name} not supported during training.')

            assert points.size(1) == npoints
            points = train_transforms(points)

            # Add noise if required
            if config.ADD_NOISE.BOOL:
                noise_level = np.random.uniform(0, 0.04)
                points, labels = add_noise_in_sphere(points, N=700)
                loss = model(points, labels, denoise=True)
            else:
                loss = model(points)

            # Backpropagation
            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            # Update running loss
            total_train_loss += loss.item()

            # Measure batch processing time
            batch_time.update(time.time() - batch_start_time)

            # Log training progress periodically
            # if idx % 20 == 0:
            #     print_log(f"[Epoch {epoch}/{config.max_epoch}][Batch {idx + 1}/{n_batches}] "
            #               f"BatchTime = {batch_time.avg:.3f}s, DataTime = {data_time.avg:.3f}s, "
            #               f"Loss = {loss.item():.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}",
            #               logger=logger)

        # Adjust learning rate
        scheduler.step()

        # Compute average training loss
        avg_train_loss = total_train_loss / n_batches
        print(
            f"Epoch [{epoch}/{config.max_epoch}] - Avg Train Loss: {avg_train_loss:.4f}, Time: {time.time() - epoch_start_time:.2f}s")

        # Save model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
        }, checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

    print("Training complete.")


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    (_, extra_train_dataloader)  = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        print('in start ckpts, loading model..')
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()

    if config.ADD_NOISE.FREEZE:
        # Freeze decoder parameters
        print('Freezing Decoder Parameters...')
        for param in base_model.module.MAE_decoder.parameters():
            param.requires_grad = False

        print('Freezing Encoder Layers except the last two...')
        # Freeze all encoder layers except the last two
        for i, block in enumerate(base_model.module.MAE_decoder.blocks):
            if i < len(base_model.module.MAE_decoder.blocks) - 2:
                for param in block.parameters():
                    param.requires_grad = False

    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    # base_model.zero_grad()
    save_img = False

# Move to new function to train classification net only
    if config.ADD_NOISE.CLASSIFICATION:
        print('Train CLASSIFICATION denoiser..')
        if config.ADD_NOISE.BOOL:
            # Freeze the encoder parameters
            print('Freezing Encoder Parameters...')
            for param in base_model.module.MAE_encoder.parameters():
                param.requires_grad = False
        # train_denoiser(config, args=args, encoder=base_model.module.MAE_encoder, train_dataloader=train_dataloader, start_epoch=start_epoch, logger=logger, checkpoint_dir='checkpoints')
    else:
        print('Train GENERATIVE denoiser..')

        # ________________________________________________________________________________
        # Subset creation (get the first `subset_size` batches)
        subset_size = 1
        train_subset = []
        for idx, batch in enumerate(train_dataloader):
            train_subset.append(batch)
            if idx + 1 == subset_size:
                break
        train_dataloader = train_subset
        # ________________________________________________________________________________
        lossList = []
        valLossList = []
        accuracyList = []
        valAccuracyList = []
        ratioList = []

        # del base_model.MAE_decoder  # Free memory after loading

        for epoch in range(start_epoch, config.max_epoch + 1):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            base_model.train()

            epoch_start_time = time.time()
            batch_start_time = time.time()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter(['Loss'])
            accuracies = AverageMeter(['Loss'])

            num_iter = 0

            save_img = False

            base_model.train()  # set model to training mode
            n_batches = len(train_dataloader) # was: train_dataloader
            # noise settings

            for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader): # was: train_dataloader
                num_iter += 1
                # print(model_ids[0])
                # print(taxonomy_ids[0])
                n_itr = epoch * n_batches + idx
                data_time.update(time.time() - batch_start_time)
                npoints = config.dataset.train.others.npoints
                dataset_name = config.dataset.train._base_.NAME
                if dataset_name == 'ShapeNet':
                    points = data.cuda()
                elif dataset_name == 'ModelNet':
                    points = data[0].cuda()
                    points = misc.fps(points, npoints)
                else:
                    raise NotImplementedError(f'Train phase do not support {dataset_name}')

                # print(f'original points shape: {points.shape}')
                assert points.size(1) == npoints
                points = train_transforms(points)
                if config.ADD_NOISE.BOOL:
                    # create noise
                    # points, labels = perturb_points(points, N=256, std=0.04)
                    points_GT = points.clone()
                    points, labels = add_noise_in_sphere(points, N=700)
                    loss, accuracy, filtered_points, filtered_labels, all_vis_points = base_model(pts=points, labels=labels)
                    if epoch % 10 == 0 and not save_img:
                        # plot_filtered_points(filtered_points, filename=os.path.join('training_vis', f'plot_training_epoch_{epoch}'))
                        plot_neighborhood_lables(filtered_points, filtered_labels, all_vis_points, filename=os.path.join('training_vis', f'plot_training_epoch_{epoch}'))
                        plot_filtered_points(neighborhood=points_GT[0], filename=f'original_PC_epoch_{epoch}.png')
                        save_img = True
                    if epoch % 10 == 0:
                        # plot_filtered_points(filtered_points, filename=os.path.join('training_vis', f'plot_training_epoch_{epoch}'))
                        # print(f'Ratio of points inside the shape is: {percentage_noise_inside_shape(filtered_points, filtered_labels)}')
                        save_img = True
                    # loss, gt_noise, reconstruction = base_model(points, labels)
                    # if epoch % 10 == 0 and not save_img:
                    #     save_training_img(epoch, points, gt_noise, reconstruction, config.ADD_NOISE.BOOL)
                    #     save_img = True
                    #     print('saved img')
                else:
                    print("Pre-Train ViT")
                    # points, labels = add_noise_in_sphere(points, N=700)
                    loss = base_model(points)
                    accuracy = 0
                    # loss, gt, full_center, reconstruction = base_model(points)
                    # if epoch % 10 == 0 and taxonomy_ids[0] == "03211117" and "6a85470c071da91a73c24ae76518fe62":
                    #     save_training_img(epoch, gt, full_center, reconstruction)
                    #     save_img = True
                    #     print('saved img')
                try:
                    loss.backward()
                    # print("Using one GPU")
                except:
                    loss = loss.mean()
                    loss.backward()
                    # print("Using multi GPUs")

                # forward
                if num_iter == config.step_per_update:
                    num_iter = 0
                    optimizer.step()
                    base_model.zero_grad()

                if args.distributed:
                    loss = dist_utils.reduce_tensor(loss, args)
                    losses.update([loss.item()*1000])
                    accuracies.update([accuracy])
                else:
                    losses.update([loss.item()*1000])
                    accuracies.update([accuracy])


                if args.distributed:
                    torch.cuda.synchronize()


                if train_writer is not None:
                    train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                    train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


                batch_time.update(time.time() - batch_start_time)
                batch_start_time = time.time()

                if idx % 20 == 0:
                    print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                                (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                                ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
            if isinstance(scheduler, list):
                for item in scheduler:
                    item.step(epoch)
            else:
                scheduler.step(epoch)
            epoch_end_time = time.time()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
            print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
                 optimizer.param_groups[0]['lr']), logger = logger)

            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
            if epoch % 25 ==0 and epoch >=250:
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args,
                                        logger=logger)
            lossList.append(losses.avg())
            # print(f'accuracy.item(): {accuracy.item()} type {type(accuracy.item())}')
            # accuracyList.append(accuracy.item())
            accuracyList.append(accuracies.avg())
            # if (config.max_epoch -accuracy.item() epoch) < 10:
            #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
            # validation here (at end of epoch)
            # valLoss, valAcc = validate_classification(base_model, test_dataloader, epoch, args, config, logger)
            # valLossList.append(valLoss)
            # valAccuracyList.append(valAcc)
        # plot_training_curve_loss({'Train Loss': lossList, 'Validation loss': valLossList})
        # plot_training_curve_acc({'Train Accuracy': accuracyList, 'Validation Accuracy': valAccuracyList})
        # Avg_ratio_of_noise_points_inside_shape = sum(ratioList) / len(ratioList)
        # print(f'Avg ratio of noise points inside shape: {Avg_ratio_of_noise_points_inside_shape}')
        if train_writer is not None:
            train_writer.close()
        if val_writer is not None:
            val_writer.close()

def validate_classification(base_model, test_dataloader, epoch, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    losses = AverageMeter(['Loss'])
    accuracies = AverageMeter(['Accuracy'])

    for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
        npoints = config.dataset.train.others.npoints
        dataset_name = config.dataset.train._base_.NAME
        if dataset_name == 'ShapeNet':
            points = data.cuda()
        elif dataset_name == 'ModelNet':
            points = data[0].cuda()
            points = misc.fps(points, npoints)
        else:
            raise NotImplementedError(f'Train phase do not support {dataset_name}')

        # print(f'original points shape: {points.shape}')
        assert points.size(1) == npoints
        points = train_transforms(points)
        if points.size(0) < 128:
            pass
        if config.ADD_NOISE.BOOL:
            points, labels = add_noise_in_sphere(points[:128, :, :], N=700) # create noise
            # print(f'labels shape":: {labels.shape}')
            # print(f'points shape":: {points.shape}')
            if labels.shape[0] < 128:
                print('Entered pass')
                # pass
            else:
                loss, accuracy, _, _, _ = base_model(points, labels)
        else:
            loss = base_model(points[:128, :, :])
        losses.update([loss.item()*1000])
        accuracies.update([accuracy])
        # break  # Just for testing
        # try:
        #     print(f'Validation loss at epoch {epoch}: {losses.avg()}')
        #     return losses.avg(), accuracy
        # finally:
        #     print(f'Problem in calling losses.avg() in epoch {epoch}')
        #     print(f'losses val: {losses.val()}')
        #     return losses.val(), accuracy
    return losses.avg(), accuracies.avg()


def validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_features = []
    test_label = []

    train_features = []
    train_label = []
    npoints = config.dataset.train.others.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(extra_train_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            train_features.append(feature.detach())
            train_label.append(target.detach())

        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)
            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            test_features.append(feature.detach())
            test_label.append(target.detach())


        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(), test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch,svm_acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', svm_acc, epoch)

    return Acc_Metric(svm_acc)


def test_net():
    pass
