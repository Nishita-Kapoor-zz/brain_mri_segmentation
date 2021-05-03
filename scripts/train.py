from utils import warmup_lr_scheduler, dice_coef_metric
import numpy as np
import torch
from utils import create_folder, save_checkpoint
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_model(args, model, train_loader, val_loader, loss, optimizer, device, threshold=0.5):
    print('Training: ' + args.model + ' Lr_Scheduler: ' + str(args.lr_scheduler))
    best_dice = 0

    logs_path = "./output/logs/" + str(args.run_name)
    checkpoint_path = "./output/checkpoints/"
    create_folder(logs_path)
    create_folder(checkpoint_path)

    tb_writer = SummaryWriter(logs_path)

    for epoch in tqdm(range(1, (args.num_epochs + 1))):
        model.train()  # Enter train mode

        losses = []
        train_total_dice = []

        if args.lr_scheduler:
            warmup_factor = 1.0 / 100
            warmup_iters = min(100, len(train_loader) - 1)
            lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

        for i_step, (image, mask) in enumerate(train_loader):
            image = image.to(device)
            mask = mask.to(device)

            outputs = model(image)

            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0

            train_dice = dice_coef_metric(out_cut, mask.data.cpu().numpy())

            train_loss = loss(outputs, mask)

            losses.append(train_loss.item())
            train_total_dice.append(train_dice)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if args.lr_scheduler:
                lr_scheduler.step()

        train_mean_loss = np.array(losses).mean()
        train_mean_dice = np.array(train_total_dice).mean()
        tb_writer.add_scalar("Train loss/epoch", train_mean_loss, epoch)
        tb_writer.add_scalar("Train DICE/epoch", train_mean_dice, epoch)
        print('[epoch %d], [iter %d / %d], [train loss %.5f], [train dice %.5f]' % (
            epoch, i_step + 1, len(train_loader), train_mean_loss, train_mean_dice))

        # Validation

        model.eval()
        val_total_dice = 0

        with torch.no_grad():
            for i_step, (images, target) in enumerate(val_loader):
                images = images.to(device)
                target = target.to(device)

                outputs = model(images)

                out_cut = np.copy(outputs.data.cpu().numpy())
                out_cut[np.nonzero(out_cut < threshold)] = 0.0
                out_cut[np.nonzero(out_cut >= threshold)] = 1.0

                val_dice = dice_coef_metric(out_cut, target.data.cpu().numpy())
                val_total_dice += val_dice

        val_mean_dice = val_total_dice / i_step
        tb_writer.add_scalar("Val DICE/epoch", val_mean_dice, epoch)

        print('------------------------------------------------------------')
        print('[epoch %d], [val DICE %.5f]' % (epoch, val_mean_dice))
        print('------------------------------------------------------------')

        if val_mean_dice > best_dice:
            # Save model
            save_checkpoint(path=checkpoint_path + "checkpoint_" + str(args.run_name) + ".pth", model=model,
                            epoch=epoch, optimizer=optimizer)
            best_dice = val_mean_dice

            print('*****************************************************')
            print('best record: [epoch %d], [val dice %.5f]' % (epoch, val_mean_dice))
            print('*****************************************************')

    tb_writer.close()
