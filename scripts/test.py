from utils import warmup_lr_scheduler, dice_coef_metric
import numpy as np
import torch
from utils import load_checkpoint
from tqdm import tqdm
import cv2
from os.path import join, exists
import matplotlib.pyplot as plt
from data.dataset import image_transforms
from utils import plot_plate_overlap


def predict_single(args,  model, device, threshold=0.5):

    image_path = args.image_path
    mask_path = image_path.split('.')[0] + '_mask.tif'

    image_original = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)

    augmented = image_transforms(image=image_original, mask=mask)

    image = augmented['image'].unsqueeze(0).to(device)
    mask = augmented['mask'].to(device)

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
    pred = model(image)
    pred = pred.detach().cpu().numpy()

    # pred with threshold
    pred_t = np.copy(pred)[0, 0, :, :]
    pred_t[np.nonzero(pred_t < threshold)] = 0.0
    pred_t[np.nonzero(pred_t >= threshold)] = 1.0  # 255.0

    dice = dice_coef_metric(pred_t, mask.data.cpu().numpy())

    # plot
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    ax[0, 0].imshow(image_original)
    ax[0, 0].set_title("Image")
    ax[0, 1].imshow(mask.detach().cpu().numpy()[0, :, :])
    ax[0, 1].set_title("Ground Truth")
    ax[1, 0].imshow(pred[0, 0, :, :], cmap="jet")
    ax[1, 0].set_title("Prediction probabilities")
    ax[1, 1].imshow(pred_t)
    ax[1, 1].set_title("Prediction with threshold=0.3")
    plt.suptitle("Dice Coeffecient: {}".format(dice.round(3)), fontsize=20)
    # plt.show()
    plt.savefig("images/prediction.png")


def evaluate(args, model, test_loader, device, threshold=0.5):
    print('Evaluating: ' + args.model + ' Lr_Scheduler: ' + str(args.lr_scheduler))

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
    return test_mean_dice


def create_gif(args, model, dataloader, device, threshold):
    print('Creating GIF: ')

    checkpoint_path = "./output/checkpoints/checkpoint_" + str(args.run_name) + ".pth"
    #checkpoint = load_checkpoint(path=checkpoint_path, model=model)

    #model = checkpoint["model"]
    model = model.to(device)

    model.eval()

    with torch.no_grad():
        count = 0
        for i_step, (images, target) in tqdm(enumerate(dataloader)):
            images = images.to(device)
            target = target.to(device)

            #prediction = model(images)

            #prediction = prediction.detach().cpu().numpy()
            images = images.detach().cpu().numpy()
            target = target.detach().cpu().numpy()

            # pred with threshold
            prediction[np.nonzero(prediction < threshold)] = 0.0
            prediction[np.nonzero(prediction >= threshold)] = 255.0  # 1.0
            prediction = prediction.astype("uint8")

            # overlap
           # original_img = cv2.resize(cv2.imread(test_sample[1]), (128, 128))

            _, thresh_gt = cv2.threshold(target, 127, 255, 0)
            _, thresh_p = cv2.threshold(prediction, 127, 255, 0)
            contours_gt, _ = cv2.findContours(thresh_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_p, _ = cv2.findContours(thresh_p, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            overlap_img = cv2.drawContours(images, contours_gt, 0, (0, 255, 0), 1)
            overlap_img = cv2.drawContours(overlap_img, contours_p, 0, (255, 36, 0), 1)  # 255,0,0

            plt.imshow(overlap_img)
            plt.show()


            #prediction_overlap.append(overlap_img)

        for i in range(5, 105 + 5, 5):
            pred_overlap_5x1.append(np.hstack(np.array(prediction_overlap[i - 5:i])))
        for i in range(3, 21 + 3, 3):
            pred_overlap_5x3.append(np.vstack(pred_overlap_5x1[i - 3:i]))

        title = "Predictions of " + str(args.model)
        for num, batch in enumerate(pred_overlap_5x3):
            plot_plate_overlap(batch, title, num)



        print(count)


def batch_preds_overlap(args, model, samples, device, threshold=0.3):
    """
    Computes prediction on the dataset

    Returns: list with images overlapping with predictions

    """
    prediction_overlap = []
    pred_overlap_5x1 = []
    pred_overlap_5x3 = []

    for test_sample in samples:

        image_original = cv2.imread(test_sample[1])
        mask = cv2.imread(test_sample[2], 0)

        augmented = image_transforms(image=image_original, mask=mask)

        image = augmented['image'].unsqueeze(0).to(device)
        ground_truth = cv2.resize(cv2.imread(test_sample[2], 0), (128, 128)).astype("uint8")

        # check if checkpoint available and load
        checkpoint_path = "../output/checkpoints/checkpoint_" + str(args.run_name) + ".pth"
        if exists(checkpoint_path):
            checkpoint = load_checkpoint(path=checkpoint_path, model=model)
        else:
            raise AssertionError("Checkpoint doesn't exist, please train model first")

        model = checkpoint["model"]

        model = model.to(device)
        model.eval()

        # pred
        pred = model(image)
        pred = pred.detach().cpu().numpy()[0, 0, :, :]

        # pred with threshold
        pred[np.nonzero(pred < threshold)] = 0.0
        pred[np.nonzero(pred >= threshold)] = 255.0  #1.0
        pred = pred.astype("uint8")

        # overlap
        original_img = cv2.resize(cv2.imread(test_sample[1]), (128, 128))

        _, thresh_gt = cv2.threshold(ground_truth, 127, 255, 0)
        _, thresh_p = cv2.threshold(pred, 127, 255, 0)
        contours_gt, _ = cv2.findContours(thresh_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_p, _ = cv2.findContours(thresh_p, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        overlap_img = cv2.drawContours(original_img, contours_gt, 0, (0, 255, 0), 1)
        overlap_img = cv2.drawContours(overlap_img, contours_p, 0, (255, 36, 0), 1)  # 255,0,0
        prediction_overlap.append(overlap_img)

    for i in range(5, 105 + 5, 5):
        pred_overlap_5x1.append(np.hstack(np.array(prediction_overlap[i - 5:i])))
    for i in range(3, 21 + 3, 3):
        pred_overlap_5x3.append(np.vstack(pred_overlap_5x1[i - 3:i]))

    title = "Predictions of " + str(args.model)
    for num, batch in enumerate(pred_overlap_5x3):
        plot_plate_overlap(batch, title, num)

    print('All images created!')
