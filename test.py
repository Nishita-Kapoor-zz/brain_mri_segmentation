


def predict():
    # image
    test_sample = test_df[test_df["diagnosis"] == 1].sample(1).values[0]
    image = cv2.resize(cv2.imread(test_sample[1]), (128, 128))

    # mask
    mask = cv2.resize(cv2.imread(test_sample[2]), (128, 128))

    # pred
    pred = torch.tensor(image.astype(np.float32) / 255.).unsqueeze(0).permute(0, 3, 1, 2)
    pred = rx50(pred.to(device))
    pred = pred.detach().cpu().numpy()[0, 0, :, :]

    # pred with tshd
    pred_t = np.copy(pred)
    pred_t[np.nonzero(pred_t < 0.3)] = 0.0
    pred_t[np.nonzero(pred_t >= 0.3)] = 255.  # 1.0
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


def evaluate():

    model.eval()
    valloss = 0

    with torch.no_grad():
        for i_step, (data, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)
            # prediction = model(x_gpu)

            outputs = model(data)
            # print("val_output:", outputs.shape)

            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0

            picloss = dice_coef_metric(out_cut, target.data.cpu().numpy())
            valloss += picloss

        # print("Threshold:  " + str(threshold) + "  Validation DICE score:", valloss / i_step)

    return valloss / i_step
