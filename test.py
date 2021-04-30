from utils import warmup_lr_scheduler, dice_coef_metric
import numpy as np
import torch
from utils import load_checkpoint
from tqdm import tqdm
import cv2
from os.path import join, exists
import matplotlib.pyplot as plt


def predict(args, model, device, threshold=0.5):

    image_path = args.image_path
    mask_path = join(image_path.split('.')[0], '_mask.tif')
    # image
    # test_sample = test_df[test_df["diagnosis"] == 1].sample(1).values[0]
    image = cv2.resize(cv2.imread(image_path), (128, 128))

    # mask
    mask = cv2.resize(cv2.imread(mask_path), (128, 128))

    # check if checkpoint available and load
    checkpoint_path = "./output/checkpoints/checkpoint_" + str(args.run_name) + ".pth"
    if exists(checkpoint_path):
        checkpoint = load_checkpoint(path=checkpoint_path, model=model)
    else:
        raise AssertionError("Checkpoint doesn't exist, please train model first")

    model = checkpoint["model"]
    model = model.to(device)
    model.eval()

    # pred
    pred = torch.tensor(image.astype(np.float32) / 255.).unsqueeze(0).permute(0, 3, 1, 2)
    pred = model(pred.to(device))
    pred = pred.detach().cpu().numpy()[0, 0, :, :]

    # pred with threshold
    pred_t = np.copy(pred)
    pred_t[np.nonzero(pred_t < threshold)] = 0.0
    pred_t[np.nonzero(pred_t >= threshold)] = 255.  # 1.0
    pred_t = pred_t.astype("uint8")

    # plot
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    ax[0, 0].imshow(image)
    ax[0, 0].set_title("image")
    ax[0, 1].imshow(mask)
    ax[0, 1].set_title("mask")
    ax[1, 0].imshow(pred)
    ax[1, 0].set_title("prediction")
    ax[1, 1].imshow(pred_t)
    ax[1, 1].set_title("prediction with threshold")
    plt.show()


def evaluate(args, model, test_loader, device, threshold=0.5):

    checkpoint_path = "./output/checkpoints/checkpoint_" + str(args.run_name) + ".pth"
    checkpoint = load_checkpoint(path=checkpoint_path, model=model)

    model = checkpoint["model"]
    model = model.to(device)

    model.eval()
    test_total_dice = 0

    with torch.no_grad():
        for i_step, (images, target) in tqdm(enumerate(test_loader)):
            images = images.to(device)
            target = target.to(device)

            outputs = model(images)

            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0

            test_dice = dice_coef_metric(out_cut, target.data.cpu().numpy())
            test_total_dice += test_dice

        test_mean_dice = test_total_dice / i_step
        print(f"""Mean dice of the test images - {np.around(test_mean_dice, 2) * 100}%""")


