import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
import torch


class Segmentor:
    def __init__(self, args):
        # create segmentation model with pretrained encoder
        self.model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            classes=1,
            activation="sigmoid",
        )
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn("resnet50", "imagenet")
        self.loss = utils.losses.DiceLoss()
        self.metrics = [
            utils.metrics.IoU(threshold=0.5),
            utils.metrics.Accuracy(threshold=0.5),
            utils.metrics.Precision(),
            utils.metrics.Recall(),
        ]
        self.device = args.device
        # self.optimizer = torch.optim.Adam([
        #     dict(params=self.model.parameters(), lr=0.0001),
        # ])

        # self.train_epoch = utils.train.TrainEpoch(
        #                     self.model,
        #                     loss=self.loss,
        #                     metrics=self.metrics,
        #                     optimizer=self.optimizer,
        #                     device=args.device,
        #                     verbose=True,
        #                 )

        # self.valid_epoch = utils.train.ValidEpoch(
        #                     self.model,
        #                     loss=self.loss,
        #                     metrics=self.metrics,
        #                     device=args.device,
        #                     verbose=True,
        #                 )

    def test_model(self, path):
        self.test_model = utils.train.ValidEpoch(
            torch.load(path),
            loss=self.loss,
            metrics=self.metrics,
            device=self.device,
            verbose=True,
        )
        self.model = torch.load(path)
