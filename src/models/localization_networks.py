import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import GaussianLayer

class Conv1dMultiscaleLocalization(nn.Module):

    def __init__(self, local_max_thr=0.5, apply_input_smoothing_for_local_max=False):
        super(Conv1dMultiscaleLocalization, self).__init__()

        self.local_max_thr = local_max_thr

        self._init_conv_buffers()

        self.gaussian_blur = GaussianLayer(num_channels=1, sigma=1) if apply_input_smoothing_for_local_max else None

    def init_output(self):
        pass

    @staticmethod
    def _generate_kernel(w=5):
        x = (w - 1) // 2
        k = -np.ones((1, w))
        k[0, 0:x] = -k[0, 0:x]
        k[0, x] = 0
        return k / (w - 1)

    def _init_conv_buffers(self):

        self._conv_merge_fn = lambda x: torch.max(x, dim=1, keepdim=True)[0]

        kernel_sizes = [3, 9, 15, 21, 31, 51, 65]

        self.max_kernel_size = max(kernel_sizes)

        kernel_weights = [np.pad(self._generate_kernel(i)[0], (self.max_kernel_size - i) // 2) for i in kernel_sizes]
        kernel_weights = np.stack(kernel_weights)

        kernel_cos = torch.tensor(kernel_weights, dtype=torch.float32).reshape(len(kernel_sizes), 1,
                                                                               self.max_kernel_size, 1)
        kernel_sin = torch.tensor(kernel_weights, dtype=torch.float32).reshape(len(kernel_sizes), 1, 1,
                                                                               self.max_kernel_size)

        self.register_buffer("kernel_cos", kernel_cos)
        self.register_buffer("kernel_sin", kernel_sin)


    def forward(self, C_, S_, ignore_region=None):

        def extend_shape(X):
            if X is not None:
                while len(X.shape) < 4:
                    X = X.unsqueeze(0)
            return X

        # convert to [B x C x H x W] if not already in this shape
        inputs = list(map(extend_shape, [C_, S_]))

        inputs = list(map(lambda X: X.detach() if X is not None else X, inputs))

        conv_resp = self._conv_response(*inputs)

        # SHOULD NOT use inplace so that returned conv_resp values will have negative values for backprop-gradient
        conv_resp_positive = F.relu(conv_resp.clone(), inplace=False)

        centers = self._get_local_max_indexes(conv_resp_positive, min_distance=5, threshold_abs=self.local_max_thr, input_smooth_op=self.gaussian_blur)

        if ignore_region is not None:
            ignore_region = extend_shape(ignore_region)
            centers = [(b, x, y, c) for (b, x, y, c) in centers if ignore_region[b, 0, y, x] == 0]

        # remove batch dim if input did not have it
        if len(C_.shape) < 4:
            centers = [(x, y, c) for (b, x, y, c) in centers]

        return centers, conv_resp

    def _conv_response(self, C, S):
        conv_resp = F.conv2d(C, self.kernel_cos, padding=(self.max_kernel_size // 2, 0)) + \
                    F.conv2d(S, self.kernel_sin, padding=(0, self.max_kernel_size // 2))

        # use max for merge
        conv_resp = self._conv_merge_fn(conv_resp)

        return conv_resp

    def _get_local_max_indexes(self, input_ch, min_distance, threshold_abs=0.0, input_smooth_op=None):
        """
        Return the indeces containing all peak candidates above thresholds.
        """
        input_ch_blured = input_smooth_op(input_ch) if input_smooth_op is not None else input_ch

        size = 2 * min_distance + 1
        input_max_pool = F.max_pool2d(input_ch_blured,
                                      kernel_size=(size, size),
                                      padding=(size // 2, size // 2), stride=1)

        mask = torch.eq(input_ch_blured, input_max_pool)

        mask = mask * (input_ch > threshold_abs)

        return [(b.item(), x.item(), y.item(), input_ch[b, c, y, x].item()) for (b, c, y, x) in torch.nonzero(mask)]

class Conv2dDilatedLocalization(Conv1dMultiscaleLocalization):

    def __init__(self, local_max_thr=0.5,
                 return_sigmoid=False, inner_ch=16, inner_kernel=5,
                 dilations=[1, 4, 8, 16, 32, 48], freeze_learning=False,
                 apply_input_smoothing_for_local_max=False):
        self.input_ch = 2
        self.inner_ch = inner_ch
        self.inner_kernel = inner_kernel
        self.dilations = dilations
        self.freeze_learning = False #freeze_learning
        self.relu_fn = nn.ReLU
        self.batch_norm_fn = nn.SyncBatchNorm if torch.distributed.is_initialized() else nn.BatchNorm2d

        super(Conv2dDilatedLocalization, self).__init__(local_max_thr, apply_input_smoothing_for_local_max)

        self.return_sigmoid = return_sigmoid

    def init_output(self):
        init_convs = []
        for c in self.conv_dilations:
            init_convs.extend(list(c))
        if type(self.conv_start) == torch.nn.modules.Sequential:
            init_convs.extend(list(self.conv_start))
        if type(self.conv_end) == torch.nn.modules.Sequential:
            init_convs.extend(list(self.conv_end))

        init_convs.append(self.conv_merge_fn)

        for c in init_convs:
            if type(c) in [torch.nn.modules.conv.Conv2d, torch.nn.modules.conv.ConvTranspose2d]:
                print('initialize center estimator layer with size: ', c.weight.size())

                torch.nn.init.xavier_normal_(c.weight, gain=0.05)  # gain=0.05)
                if c.bias is not None:
                    torch.nn.init.zeros_(c.bias)

    def _init_conv_buffers(self):

        use_batchnorm = True
        if use_batchnorm is False:
            self.batch_norm_fn = lambda **args: nn.Identity()

        def conv2d_block(in_ch, out_ch, kw, p, d=1, s=1):
            return [nn.Conv2d(in_ch, out_ch, kernel_size=kw, padding=p, dilation=d, stride=s, bias=not use_batchnorm),
                    self.batch_norm_fn(num_features=out_ch, track_running_stats=not self.freeze_learning),
                    self.relu_fn(inplace=True)]

        def deconv2d_block(in_ch, out_ch, kw, p, out_p, s=1):
            return [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kw, padding=p, output_padding=out_p, stride=s, bias=not use_batchnorm),
                    self.batch_norm_fn(num_features=out_ch, track_running_stats=not self.freeze_learning),
                    self.relu_fn(inplace=True)]

        self.conv_merge_fn = lambda x: torch.max(x, dim=1, keepdim=True)[0]
        start_nn = nn.Sequential(
            *conv2d_block(self.input_ch, self.inner_ch, kw=self.inner_kernel, p=self.inner_kernel // 2, s=2),
            *conv2d_block(self.inner_ch, self.inner_ch, kw=self.inner_kernel, p=self.inner_kernel // 2, s=2),
            *conv2d_block(self.inner_ch, 2 * self.inner_ch, kw=self.inner_kernel, p=self.inner_kernel // 2, s=2),
            *conv2d_block(2 * self.inner_ch, 2 * self.inner_ch, kw=self.inner_kernel, p=self.inner_kernel // 2, s=2),
        )
        seq_nn = nn.ModuleList([
            nn.Sequential(
                *conv2d_block(2 * self.inner_ch, 2 * self.inner_ch, kw=self.inner_kernel, p=self.inner_kernel // 2 * d, d=d)
            )
            for d in self.dilations
        ])
        end_nn = nn.Sequential(
            *deconv2d_block(len(self.dilations) * 2 * self.inner_ch, 2 * self.inner_ch, kw=5, p=5 // 2, out_p=3 // 2, s=2),
            *deconv2d_block(2 * self.inner_ch, 2 * self.inner_ch, kw=5, p=5 // 2, out_p=3 // 2, s=2),
            *deconv2d_block(2 * self.inner_ch, self.inner_ch, kw=5, p=5 // 2, out_p=3 // 2, s=2),
            *deconv2d_block(self.inner_ch, self.inner_ch, kw=5, p=5 // 2, out_p=3 // 2, s=2),
        )

        self.conv_start = start_nn
        self.conv_dilations = seq_nn
        self.conv_end = end_nn

        self.conv_merge_fn = nn.Conv2d(self.inner_ch, 1, kernel_size=3, padding=3 // 2)

    def _conv_response(self, C, S):

        input = [C, S]

        input = torch.cat(input, dim=1)

        c1 = self.conv_start(input)
        conv_resp = [conv_op(c1) for conv_op in self.conv_dilations]
        conv_resp = self.conv_end(torch.cat(conv_resp, dim=1))

        # use max
        conv_resp = self.conv_merge_fn(conv_resp)

        if self.return_sigmoid:
            conv_resp = torch.sigmoid(conv_resp)

        return conv_resp
