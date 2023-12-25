import typing as tp

import torch


class SAD:
    """
    SAD(Source Activity Detector)
    """

    def __init__(
            self,
            sr: int,
            window_size_in_sec: int = 6,
            overlap_ratio: float = 0.5,
            n_chunks_per_segment: int = 10,
            eps: float = 1e-5,
            gamma: float = 1e-3,
            threshold_max_quantile: float = 0.15,
            threshold_segment: float = 0.5,
    ):
        self.sr = sr
        self.n_chunks_per_segment = n_chunks_per_segment
        self.eps = eps
        self.gamma = gamma
        self.threshold_max_quantile = threshold_max_quantile
        self.threshold_segment = threshold_segment

        self.window_size = sr * window_size_in_sec
        self.step_size = int(self.window_size * overlap_ratio)

    def chunk(self, y: torch.Tensor):
        """
        Input shape: [n_channels, n_frames]
        Output shape: []
        """
        y = y.unfold(-1, self.window_size, self.step_size)   # shape: [n_channels, n_segment, window_size]
        y = y.chunk(self.n_chunks_per_segment, dim=-1)
        y = torch.stack(y, dim=-2)                           # shape: [n_channels, n_segment, n_chunks, chunks_size]
        return y

    @staticmethod
    def calculate_rms(y: torch.Tensor):
        """
        input shape: [n_channels, n_segment, n_chunks, chunks_size]
        output shape: [n_channels, n_segment, n_chunks, 1]
        """
        y = torch.mean(torch.square(y), dim=-1, keepdim=True)
        y = torch.sqrt(y)
        return y

    def calculate_thresholds(self, rms: torch.Tensor):
        """
        input shape: [n_channels, n_segment, n_chunks, 1]
        output shape: []
        """
        rms[rms == 0.] = self.eps              # [n_channels, n_segment, n_chunks, 1]
        rms_threshold = torch.quantile(
            rms,
            self.threshold_max_quantile,
            dim=-2,                            # 计算10个chunk的rms分位数
            keepdim=True,
        )                                      # [n_channels, n_segment, 1, 1]
        rms_threshold[rms_threshold < self.gamma] = self.gamma     # 小于gamma的都设为gamma
        rms_percentage = torch.mean(           # 计算每个segment中有多少个chunk小于这个阈值
            (rms > rms_threshold).float(),
            dim=-2,
            keepdim=True,
        )                                      # [n_channels, n_segment, 1, 1]
        rms_mask = torch.all(rms_percentage > self.threshold_segment, dim=0).squeeze()    # dim=0表示沿着该维度进行逻辑与操作，只有当两个通道的值都大于该值，才认为合法[n_segment, 1, 1] >>>> [n_segment]
        return rms_mask     # 针对segment的rms mask

    def calculate_salient(self, y: torch.Tensor, mask: torch.Tensor):
        """
        """
        y = y[:, mask, ...]          # 丢弃mask = False的segment
        C, D1, D2, D3 = y.shape
        y = y.view(C, D1, D2*D3)     # 将每个segment中的chunk合并到一起
        return y

    def __call__(                    # 使实例对象可以像函数一样被调用，并调用该方法
            self,
            y: torch.Tensor,
            segment_saliency_mask: tp.Optional[torch.Tensor] = None
    ):
        """
        Stacks signal into segments and filters out silent segments.
        :param y: Input signal.
            Shape [n_channels, n_frames]
               segment_saliency_mask: Optional precomputed mask
            Shape [n_channels, n_segments, 1, 1]
        :return: Salient signal folded into segments of length 'self.window_size' and step 'self.step_size'.
            Shape [n_channels, n_segments, frames_in_segment]
        """
        y = self.chunk(y)
        rms = self.calculate_rms(y)
        if segment_saliency_mask is None:
            segment_saliency_mask = self.calculate_thresholds(rms)
        y_salient = self.calculate_salient(y, segment_saliency_mask)
        return y_salient, segment_saliency_mask

    def calculate_salient_indices(
            self,
            y: torch.Tensor
    ):
        """
        Returns start indices of salient regions of audio
        """
        y = self.chunk(y)
        rms = self.calculate_rms(y)
        mask = self.calculate_thresholds(rms)
        indices = torch.arange(mask.shape[-1])[mask] * self.step_size
        return indices.tolist()


if __name__ == "__main__":
    import torchaudio

    sr = 44100
    example_path = 'example/example.mp3'

    sad = SAD(sr=sr)
    y, sr = torchaudio.load(example_path)
    y_salience = sad(y)[0]    # 相当于直接调用 sad.__call__(y)
    print(f"Initial shape: {y.shape}.\nShape after source activity detection: {y_salience.shape}")


