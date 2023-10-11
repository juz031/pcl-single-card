import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio.v3 as imageio
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import os
from skimage.morphology import erosion, dilation, closing, opening, area_closing, area_opening



def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.33, 0.33, 0.33])


def plot_from_tensors(outputs, titles, detach=False, vmin=None, vmax=None):
    plt.clf()
    plt.figure(figsize=(7 * len(outputs), 5 * len(outputs)))
    for idx, output in enumerate(outputs):
        plt.subplot(1, len(outputs), idx + 1)
        with torch.no_grad():
            _, c, new_h, new_w = output.shape  # output is [batch_size, output_channel, new_h, new_w]
            output = torch.transpose(output.squeeze(0).reshape(1, -1), 0, 1).reshape(new_h, new_w, c).squeeze(-1)
            print(output.shape)

            if detach:
                output = output.detach()
        output = output.numpy()  # convert to numpy and normalize the pixels between 0 and 1.

        if vmax:
            plt.imshow(output, cmap='gray', vmin=vmin, vmax=vmax)
        else:
            plt.imshow(output, cmap='gray')
        plt.title(titles[idx])


# function that mask out the output tensor
@torch.no_grad()
def mask_out(output_tensor, mask):
    """
  output_tensor: [:, :, output_h, output_w]
  mask: [output_h, output_w]
  """
    output_tensor[torch.from_numpy((1. - mask)).unsqueeze(0).unsqueeze(0).bool()] = 0.
    return output_tensor



class MyModel(nn.Module):
    def __init__(self, init_img):
        super(MyModel, self).__init__()
        self.img = nn.Parameter(torch.empty_like(init_img).copy_(init_img).float())

    def forward(self, objective_ix_t, objective_iy_t, ikernel_x, ikernel_y, mask_tensor):
        # convolution for calculate ix and iy for the input image
        input_ix = F.conv2d(self.img,
                            ikernel_x)  # F is an abbreviation for torch.nn.functional, have a look at the documentation for pytorch convolution operation if confused (https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html#torch.nn.functional.conv2d)
        input_iy = F.conv2d(self.img, ikernel_y)

        # Mask out the blank region to avoid confusion during optimization
        input_ix = input_ix * mask_tensor[:, :, :, :-1]
        input_iy = input_iy * mask_tensor[:, :, :-1, :]
        objective_ix_t = objective_ix_t * mask_tensor[:, :, :, :-1]
        objective_iy_t = objective_iy_t * mask_tensor[:, :, :-1, :]

        # Calculate L1 loss
        loss = ((input_ix - objective_ix_t).abs()).mean() + ((input_iy - objective_iy_t).abs()).mean()
        return loss


# train function that can use different thresholds
def train(threshold_x, threshold_y, objective_ix, objective_iy, x, mask, ikernel_x, ikernel_y, h, w, train_steps=5000):
    # thresholding
    objective_ix_t = objective_ix * (objective_ix.abs() > threshold_x)
    objective_iy_t = objective_iy * (objective_iy.abs() > threshold_y)

    # initialize the input image, we initiliaze it from a zero image
    init_img = torch.zeros_like(x.reshape(1, 1, h, w)).cuda()
    # wrap the input image that we want to optimize into a nn.Module subclass, install a optimizer to it so that it can be easily optimized
    model = MyModel(init_img).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    # prepare a tensor version of mask
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).cuda()

    # to GPU
    objective_ix_t = objective_ix_t.cuda()
    objective_iy_t = objective_iy_t.cuda()
    ikernel_x = ikernel_x.cuda()
    ikernel_y = ikernel_y.cuda()

    # start training loop
    # t = tqdm(enumerate(range(train_steps)), desc='Loss: **** ', total=train_steps, bar_format='{desc}{bar}{r_bar}')
    for _ in range(train_steps):
        model.zero_grad()  # clear the stored grad of the model parameters
        loss = model(objective_ix_t, objective_iy_t, ikernel_x, ikernel_y, mask_tensor)
        # t.set_description('Loss: %.5f ' % loss.item())
        loss.backward()
        optimizer.step()

    # calculate and normalize original/reflectant
    # mask = np.stack((mask,)*3, axis=0)
    original = torch.exp(x.reshape(1, 1, h, w).float()) / torch.exp(x.float()).max()
    refl = mask_out(torch.exp(model.img.cpu()), mask)
    # refl = torch.exp(model.img.cpu())
    refl = refl / refl.max()

    # calculate and normalize the shading image
    shading_img = mask_out(original / (refl + 1e-12), mask)  # add 1e+12 to avoid numerical issue of division
    # shading_img = original / (refl+1e-12)
    shading_img = shading_img / shading_img.max()

    # generate a comparsion plot
    # plot_from_tensors([shading_img, refl, original], ['shading', 'reflectance', 'original'], detach=True, vmax=1.,
    #                   vmin=0.)

    return model, shading_img, refl




def main():
    traindir = '/ocean/projects/cis230063p/jzhao7/imagenet_shape_10/train_0'

    for cat in tqdm(os.listdir(traindir)):
        catdir = os.path.join(traindir, cat)
        savedir = catdir.replace('train', 'shading')
        os.makedirs(savedir, exist_ok=True)
        for filename in os.listdir(catdir):
            image = imageio.imread(os.path.join(catdir, filename))
            if len(image.shape) == 3:
                img = rgb2gray(image)
            h, w = image.shape[0], image.shape[1]
            mask_path = os.path.join(catdir, filename).replace('train', 'shape').replace('JPEG', 'jpg')
            # print(mask_path)
            mask = imageio.imread(mask_path).copy()
            # print(mask.flags)

            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            element = np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]])
            mask = opening(mask, element)

            img[img == 0.] = 1.
            log_img = np.log(img)

            x = torch.from_numpy(log_img)

            with torch.no_grad():
                ikernel_x = torch.tensor([[-1., 1.]]).unsqueeze(0).unsqueeze(0).float()
                ikernel_y = torch.tensor([[-1.], [1.]]).unsqueeze(0).unsqueeze(0).float()

                # convolution
                objective_ix = F.conv2d(x.reshape(1, 1, h, w).float(),
                                        ikernel_x).detach()  # F is an abbreviation for torch.nn.functional, have a look at the documentation for pytorch convolution operation if confused (https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html#torch.nn.functional.conv2d)
                objective_iy = F.conv2d(x.reshape(1, 1, h, w).float(),
                                        ikernel_y).detach()

            threshold_x = threshold_y = 0.1

            # apply retinex to rgb channels separately
            model, shading, refl = train(threshold_x, threshold_y, objective_ix, objective_iy, x, mask,
                                         ikernel_x, ikernel_y, h, w)
            shading_img = torch.transpose(shading.squeeze(0).reshape(1, -1), 0, 1).reshape(h, w, 1).squeeze(-1).detach().numpy()
            shading_img *= 255
            shading_img= shading_img.astype(np.uint8)
            # print(shading_img.max())
            # print(shading_img.min())
            saved_path = os.path.join(catdir, filename).replace('train', 'shading')
            imageio.imwrite(saved_path, shading_img)

if __name__ == '__main__':
    main()