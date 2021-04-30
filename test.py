from utils import warmup_lr_scheduler, dice_coef_metric
import numpy as np
import torch
from utils import load_checkpoint
from tqdm import tqdm
import cv2
from os.path import join, exists
import matplotlib.pyplot as plt


def predict_single(args, model, device, threshold=0.5):

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



def batch_preds_overlap(args, model, samples, device, threshold=0.3):
    """
    Computes prediction on the dataset

    Returns: list with images overlapping with predictions

    """
    prediction_overlap = []

    for test_sample in samples:
        # sample
        image = cv2.resize(cv2.imread(test_sample[1]), (128, 128))
        image = image / 255.

        # gt
        ground_truth = cv2.resize(cv2.imread(test_sample[2], 0), (128, 128)).astype("uint8")

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
        prediction = torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2)
        prediction = model(prediction.to(device).float())
        prediction = prediction.detach().cpu().numpy()[0, 0, :, :]

        prediction[np.nonzero(prediction < threshold)] = 0.0
        prediction[np.nonzero(prediction >= threshold)] = 255.  # 1.0
        prediction = prediction.astype("uint8")

        # overlap
        original_img = cv2.resize(cv2.imread(test_sample[1]), (128, 128))

        _, thresh_gt = cv2.threshold(ground_truth, 127, 255, 0)
        _, thresh_p = cv2.threshold(prediction, 127, 255, 0)
        contours_gt, _ = cv2.findContours(thresh_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_p, _ = cv2.findContours(thresh_p, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        overlap_img = cv2.drawContours(original_img, contours_gt, 0, (0, 255, 0), 1)
        overlap_img = cv2.drawContours(overlap_img, contours_p, 0, (255, 36, 0), 1)  # 255,0,0
        prediction_overlap.append(overlap_img)

    return prediction_overlap


