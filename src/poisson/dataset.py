import torch


class Sampler(object):
    def __init__(self, config, name="sampler", **kwargs):

        self.interior_samples = config["dataset_config"]["interior_samples"]
        self.x_range = config["physical_config"]["x_range"]

        device_ids = config["model_config"]["device_ids"]
        self.device = torch.device(
            "cuda:{:d}".format(device_ids[0]) if torch.cuda.is_available() else "cpu"
        )

    def interior(self):
        x = self.x_range[0] + torch.rand((self.interior_samples, 1)).to(self.device) * (
            self.x_range[-1] - self.x_range[0]
        )
        return x
