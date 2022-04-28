import os

import collections
import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt


class Visualizer:

    def __init__(self, keys, to_file_only=False):
        self.wins = {k:None for k in keys}

        self.to_file_only = to_file_only

    def display(self, image, key, title=None, denormalize_args=None, autoadjust_figure_size=False, **kwargs):

        n_images = len(image) if isinstance(image, (list, tuple)) else 1

        if self.wins[key] is None:
            n_cols = int(np.maximum(1,np.ceil(np.sqrt(n_images))))
            n_rows = int(np.maximum(1,np.ceil(n_images/n_cols)))
            self.wins[key] = plt.subplots(ncols=n_cols,nrows=n_rows)

        fig, ax = self.wins[key]

        if isinstance(ax, collections.Iterable):
            ax = ax.reshape(-1)
            n_axes = len(ax)
        else:
            n_axes= 1
        assert n_images <= n_axes

        if n_images == 1:
            ax.cla()
            ax.set_axis_off()
            ax.imshow(self.prepare_img(image, denormalize_args), **kwargs)
            max_size = image.shape[-2:]
        else:
            max_size = 0,0
            for i,ax_i in enumerate(ax):
                ax_i.cla()
                ax_i.set_axis_off()
                if i < n_images:
                    ax_i.imshow(self.prepare_img(image[i], denormalize_args), **kwargs)
                    max_size = max(max_size[0], image[i].shape[-2]), max(max_size[1], image[i].shape[-1])

        fig.subplots_adjust(top=1.0, bottom=0.00, left=0.01, right=0.99, wspace=0.01,hspace=0.01)

        if title is not None:
            fig.suptitle(title)

        if autoadjust_figure_size:
            if min(max_size) > 2*1024:
                f = (2*1024.0)/min(max_size)
                max_size = max_size[0]*f, max_size[1]*f

            n_cols, n_rows = fig.axes[0].get_subplotspec().get_gridspec().get_geometry()
            fig.set_size_inches(n_rows*max_size[1]/100, n_cols*max_size[0]/100)

        if True:
            plt.draw()
            self.mypause(0.001)

        return fig,ax

    def display_opencv(self, image, key, save=None, title=None, denormalize_args=None, autoadjust_figure_size=False,
                       plot_fn = None, image_colormap=None, **kwargs):
        def prepare_img_cv(im, colormap=cv2.COLORMAP_PARULA):
            im = self.prepare_img(im, denormalize_args)

            if len(im.shape) == 3 and im.shape[-1] == 3:
                # convert RGB to BGR
                im = im[:,:,[2,1,0]]

            if len(im.shape) == 2 and im.dtype.kind != 'u':
                im = (((im-im.min()) / (im.max() - im.min())) * 255).astype(np.uint8)
                im = cv2.applyColorMap(im, colormap)
            elif im.dtype.kind != 'u':
                im = (im * 255).astype(np.uint8)

            im = np.ascontiguousarray(im, dtype=np.uint8)

            if plot_fn is not None:
                im = plot_fn(im)

            return im

        if not isinstance(image, (list, tuple)):
            image = [image]

        n_images = len(image)

        n_cols = int(np.maximum(1, np.ceil(np.sqrt(n_images))))
        n_rows = int(np.maximum(1, np.ceil(n_images / n_cols)))

        if image_colormap is None:
            image_colormap = [cv2.COLORMAP_PARULA]*n_images
        elif not isinstance(image_colormap, (list, tuple)):
            image_colormap = [image_colormap]

        # prepare images
        I = [prepare_img_cv(I,cmap) for I,cmap in zip(image,image_colormap)]

        # convert to grid with n_cols and n_rows
        if len(I) < n_cols*n_rows:
            I += [np.ones_like(I[0])*255] * (n_cols*n_rows - len(I))

        I = np.concatenate([np.concatenate(I[i*n_cols:(i+1)*n_cols], axis=1) for i in range(n_rows)],axis=0)

        return I



    @staticmethod
    def prepare_img(image, denormalize_args=None):
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, torch.Tensor):
            image.squeeze_()
            image = image.numpy()
            if denormalize_args is not None:
                denorm_mean,denorm_std = denormalize_args

                image = (image * denorm_std.numpy().reshape(-1,1,1)) +  denorm_mean.numpy().reshape(-1,1,1)


        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] in {1, 3}:
                image = image.transpose(1, 2, 0)
            return image

    @staticmethod
    def mypause(interval):
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return


    def display_centerdir_predictions(self, im, output, pred_heatmap, gt_list, is_difficult_gt,
                                      pred_list, base, save_dir=None, autoadjust_figure_size=False):
        if self.to_file_only:
            self._display_centerdir_predictions_opencv(im, output, pred_heatmap,
                                                       gt_list, is_difficult_gt, pred_list, base, save_dir,
                                                       autoadjust_figure_size)
            return
        def plot_prediction_markers(ax_, **plot_args):
            if len(pred_list) > 0:
                pred_list_true = pred_list[pred_list[:, 2] > 0, :]
                pred_list_false = pred_list[pred_list[:, 2] <= 0, :]

                ax_.plot(pred_list_true[:, 0], pred_list_true[:, 1], 'gx', **plot_args)
                ax_.plot(pred_list_false[:, 0], pred_list_false[:, 1], 'rx', **plot_args)


        fig_img, ax = self.display(im.cpu(), 'image', force_draw=False, autoadjust_figure_size=autoadjust_figure_size)

        plot_prediction_markers(ax, markersize=10, markeredgewidth=2)

        if len(gt_list[is_difficult_gt == 0]) > 0:
            ax.plot(gt_list[is_difficult_gt == 0, 1], gt_list[is_difficult_gt==0, 0], 'g.',
                    markersize=5, markeredgewidth=0.2, markerfacecolor=(0,1,0,1), markeredgecolor=(0, 0, 0, 1))
        if len(gt_list[is_difficult_gt != 0]) > 0:
            ax.plot(gt_list[is_difficult_gt!=0, 1], gt_list[is_difficult_gt != 0, 0], 'y.',
                    markersize=5, markeredgewidth=0.2, markerfacecolor=(1,1,0,1), markeredgecolor=(0, 0, 0, 1))

        fig_centers, ax = self.display([(output[1]).detach().cpu(),
                                        (output[0]).detach().cpu(),
                                        pred_heatmap.detach().cpu()], 'centers', force_draw=False,
                                       autoadjust_figure_size=autoadjust_figure_size)
        for ax_i in ax:
            plot_prediction_markers(ax_i, markersize=4, markeredgewidth=1)

        if save_dir is not None:
            fig_img.savefig(os.path.join(save_dir, '%s_0.img.png' % base))
            fig_centers.savefig(os.path.join(save_dir, '%s_1.centers.png' % base))


    def _display_centerdir_predictions_opencv(self, im, output, pred_heatmap, gt_list, is_difficult_gt,
                                              pred_list, base, save_dir=None, autoadjust_figure_size=False):
        from functools import partial

        def plot_prediction_markers(img, gt=False, predictions_args=dict(), bbox_args=dict()):
            predictions_args_ = dict(markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
            predictions_args_.update(predictions_args)
            bbox_args_ = dict(thickness=3)
            bbox_args_.update(bbox_args)
            if len(pred_list) > 0:
                pred_list_true = pred_list[pred_list[:, 2] > 0, :]
                pred_list_false = pred_list[pred_list[:, 2] <= 0, :]

                for p in pred_list_true: cv2.drawMarker(img, (int(p[0]),int(p[1])), color=(0,255,0), **predictions_args_)
                for p in pred_list_false: cv2.drawMarker(img, (int(p[0]),int(p[1])), color=(0,0,255), **predictions_args_)
            if gt:
                for i,p in enumerate(gt_list):
                    cv2.circle(img, (int(p[1]), int(p[0])), radius=3, color=(0, 255, 0) if is_difficult_gt[i] == 0 else (0, 255, 255), thickness=-1)
                    cv2.circle(img, (int(p[1]), int(p[0])), radius=3, color=(0, 0, 0), thickness=1)

            return img

        fig_img = self.display_opencv(im.cpu(), 'image', autoadjust_figure_size=autoadjust_figure_size,
                                      plot_fn=partial(plot_prediction_markers, gt=True))

        fig_centers = self.display_opencv([(output[1]).detach().cpu(),
                                           (output[0]).detach().cpu(),
                                           torch.atan2(output[0],output[1]).detach().cpu(),
                                           pred_heatmap.detach().cpu()], 'centers',
                                          autoadjust_figure_size=autoadjust_figure_size,
                                          plot_fn=partial(plot_prediction_markers, predictions_args=dict(thickness=1)),
                                          image_colormap=[cv2.COLORMAP_PARULA,cv2.COLORMAP_PARULA,cv2.COLORMAP_HSV,cv2.COLORMAP_PARULA])

        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '%s_0.img.png' % base), fig_img)
            cv2.imwrite(os.path.join(save_dir, '%s_1.centers.png' % base), fig_centers)

    def display_centerdir_training(self, im, output, center_conv_resp=None, centerdir_gt=None, gt_centers_dict=None, gt_difficult=None,
                                   plot_batch_i=0, device=None, denormalize_args=None):
        with torch.no_grad():
            gt_list = np.array([c for k, c in gt_centers_dict[plot_batch_i].items()]) if gt_centers_dict is not None else []
            is_difficult_gt = np.array([False] * len(gt_list))
            if gt_difficult is not None:
                gt_difficult = gt_difficult[plot_batch_i]
                is_difficult_gt = np.array([gt_difficult[np.clip(int(c[0]),0,gt_difficult.shape[0]-1),
                                                         np.clip(int(c[1]),0,gt_difficult.shape[1]-1)].item() != 0 for c in gt_list])
            def plot_gt(ax_):
                if type(ax_) not in [list, tuple, np.ndarray]:
                    ax_ = [ax_]
                gt_list_easy, gt_list_hard = gt_list[is_difficult_gt == 0], gt_list[is_difficult_gt != 0]

                for a in ax_:
                    if len(gt_list_easy) > 0:
                        a.plot(gt_list_easy[:, 1], gt_list_easy[:, 0], 'g.', markersize=5, markeredgewidth=0.2,
                               markerfacecolor=(0, 1, 0, 1), markeredgecolor=(0, 0, 0, 1))
                    if len(gt_list_hard) > 0:
                        a.plot(gt_list_hard[:, 1], gt_list_hard[:, 0], 'y.', markersize=5, markeredgewidth=0.2,
                               markerfacecolor=(1, 1, 0, 1), markeredgecolor=(0, 0, 0, 1))


            _,ax = self.display(im[plot_batch_i].cpu(), 'image', denormalize_args=denormalize_args)
            plot_gt(ax)

            out_viz = [(output[plot_batch_i, 1]).detach().cpu(),
                       (output[plot_batch_i, 0]).detach().cpu()]
            if centerdir_gt is not None and len(centerdir_gt) > 0:
                gt_R, gt_theta, gt_sin_th, gt_cos_th = centerdir_gt[0][:, 0], centerdir_gt[0][:, 1], centerdir_gt[0][:, 2], \
                                                       centerdir_gt[0][:, 3]

                out_viz += [torch.abs(output[plot_batch_i, 1] - gt_cos_th[plot_batch_i, 0].to(device)).detach().cpu(),
                            torch.abs(output[plot_batch_i, 0] - gt_sin_th[plot_batch_i, 0].to(device)).detach().cpu()]
            else:
                out_viz += [(output[plot_batch_i, 3 + i]).detach().cpu() for i in
                            range(len(output[plot_batch_i, :]) - 4)]

            _, ax = self.display(out_viz, 'centers')
            plot_gt(ax)

            if center_conv_resp is not None:
                conv_centers = center_conv_resp[plot_batch_i].detach().cpu()
                if centerdir_gt is not None and len(centerdir_gt) > 0:
                    gt_center_mask = centerdir_gt[0][plot_batch_i, 5]
                    conv_centers = [conv_centers,
                                    torch.abs(center_conv_resp[plot_batch_i] - gt_center_mask.to(device)).detach().cpu()]

                _, ax = self.display(conv_centers, 'conv_centers')
                plot_gt(ax)

            if centerdir_gt is not None and len(centerdir_gt) > 0:
                _, ax = self.display((centerdir_gt[0][plot_batch_i, 0].cpu(),
                                    centerdir_gt[0][plot_batch_i, 1].cpu(),
                                    centerdir_gt[0][plot_batch_i, 2].cpu(),
                                    centerdir_gt[0][plot_batch_i, 3].cpu(),), 'centerdir_gt')
                plot_gt(ax)