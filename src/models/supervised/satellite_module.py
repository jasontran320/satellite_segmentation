import torch
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import torchmetrics

from src.models.supervised.segmentation_cnn import SegmentationCNN
from src.models.supervised.unet import UNet
from src.models.supervised.resnet_transfer import FCNResnetTransfer
from src.models.supervised.unetpp import Unet_2D
from src.models.supervised.unet3pp import NestedUNet


class ESDSegmentation(pl.LightningModule):
    def __init__(self, model_type, in_channels, out_channels,
                 learning_rate = 1e-3, model_params: dict = {}):
        '''
        Constructor for ESDSegmentation class.
        '''
        # call the constructor of the parent class
        super(ESDSegmentation, self).__init__()

        # use self.save_hyperparameters to ensure that the module will load
        self.save_hyperparameters()
        # gradescope test requirement
        self.learning_rate = learning_rate
        # store in_channels and out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        # if the model type is segmentation_cnn, initalize a unet as self.model
        if model_type == 'SegmentationCNN':
            self.model = SegmentationCNN(self.in_channels, self.out_channels)
        # if the model type is unet, initialize a unet as self.model
        elif model_type == 'UNet':
            self.model = UNet(self.in_channels, self.out_channels)
        # if the model type is fcn_resnet_transfer, initialize a fcn_resnet_transfer as self.model
        elif model_type == 'FCNResnetTransfer':
            self.model = FCNResnetTransfer(in_channels, out_channels)
        elif model_type == 'UNet++':
            # self.model = Unet_2D(self.in_channels, self.out_channels)
            self.model = NestedUNet(self.in_channels, self.out_channels)

        # initialize the accuracy metrics for the semantic segmentation task
        self.jaccard_index = torchmetrics.JaccardIndex(task = 'multiclass',
                                                       num_classes = out_channels)
        self.iou = torchmetrics.detection.IntersectionOverUnion(class_metrics = True)
        self.auc = torchmetrics.AUROC(task = 'multiclass', num_classes = out_channels)
        self.f1 = torchmetrics.F1Score(task = 'multiclass', num_classes = out_channels)
        self.acc = torchmetrics.Accuracy(task = 'multiclass', num_classes = out_channels)

    def forward(self, X):
        # evaluate self.model
        return self.model(torch.nan_to_num(X.to(torch.float32)))

    def training_step(self, batch, batch_idx):
        # get sat_img and mask from batch
        sat_img, mask = batch
        # evaluate batch
        logits = self.forward(sat_img)
        # calculate cross entropy loss
        loss = nn.functional.cross_entropy(logits, mask.long())
        self.log('Training step loss', loss)
        # return loss
        return loss

    def validation_step(self, batch, batch_idx):
        # get sat_img and mask from batch
        sat_img, mask = batch
        # evaluate batch for validation
        logits = self.forward(sat_img)
        # get the class with the highest probability
        prediction = torch.argmax(logits, dim = 1)
        # evaluate each accuracy metric and log it in wandb

        jaccard_index = self.jaccard_index(prediction, mask)
        auc = self.auc(logits, mask.long())
        f1 = self.f1(prediction, mask)
        acc = self.acc(prediction, mask)
        self.log('Jaccard Index', jaccard_index)
        self.log('AUC: Area under the Receiver Operator Curve (ROC)', auc)
        self.log('F1-Score', f1)
        self.log('Accuracy', acc)
        # return validation loss
        loss = nn.functional.cross_entropy(logits, mask.long())
        self.log('Validation loss', loss)
        return loss

    def configure_optimizers(self):
        # initialize optimizer
        optimizer = Adam(self.parameters(), lr = self.hparams.learning_rate)
        # return optimizer
        return optimizer