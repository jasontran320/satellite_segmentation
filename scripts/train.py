import sys
from argparse import ArgumentParser

from pathlib import Path
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger


sys.path.append(".")
from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.utilities import ESDConfig, PROJ_NAME


ROOT = Path.cwd()


def train(options: ESDConfig):
    # initialize wandb
    # setup the wandb logger
    wandb.init(project = PROJ_NAME)
    wandb_logger = WandbLogger(project = PROJ_NAME)
    # initialize the datamodule
    datamodule = ESDDataModule(options.processed_dir, options.raw_dir, options.batch_size,
                               options.num_workers, options.seed, options.selected_bands,
                               options.slice_size)

    # prepare the data
    datamodule.prepare_data()
    # create a model params dict to initialize ESDSegmentation
    # note: different models have different parameters
    model_params = dict()
    if options.model_type == 'SegmentationCNN':
        model_params = {'depth': options.depth,
                        'embedding_size': options.embedding_size,
                        'pool_sizes': options.pool_sizes,
                        'kernel_size': options.kernel_size,
                        }
    elif options.model_type == 'UNet':
        model_params = {'n_encoders': options.n_encoders,
                        'embedding_size': options.embedding_size}
    elif options.model_type == 'FCNResnetTransfer':
        model_params = dict()
    elif options.model_type == 'Unet++':
        model_params = dict()
    # initialize the ESDSegmentation model
    model = ESDSegmentation(options.model_type, options.in_channels, options.out_channels,
                            options.learning_rate, model_params)
    # model = ESDSegmentation.load_from_checkpoint(options.model_path)
    # Use the following callbacks, they're provided for you,
    # but you may change some of the settings
    # ModelCheckpoint: saves intermediate results for the neural network
    # in case it crashes
    # LearningRateMonitor: logs the current learning rate on weights and biases
    # RichProgressBar: nicer looking progress bar (requires the rich package)
    # RichModelSummary: shows a summary of the model before training (requires rich)
    callbacks = [
        ModelCheckpoint(
            dirpath = ROOT / "models" / options.model_type,
            filename = "{epoch}-{val_loss:.2f}-{other_metric:.2f}",
            save_top_k = 0,
            save_last = True,
            verbose = True,
            monitor = "val_loss",
            mode = "min",
            every_n_train_steps = 1000,
        ),
        LearningRateMonitor(),
        RichProgressBar(),
        RichModelSummary(max_depth = 3),
    ]

    # initialize trainer, set accelerator, devices, number of nodes, logger
    # max epochs and callbacks
    trainer = pl.Trainer(accelerator = options.accelerator,
                         devices = options.devices, num_nodes = 1,
                         logger = wandb_logger,
                         max_epochs = options.max_epochs,
                         callbacks = callbacks)

    # run trainer.fit
    trainer.fit(model = model, datamodule = datamodule)

    wandb.finish()  # remove after debugging


if __name__ == "__main__":
    # load dataclass arguments from yml file

    config = ESDConfig()
    parser = ArgumentParser()

    parser.add_argument(
        "--model_type",
        type = str,
        help = "The model to initialize.",
        default = config.model_type,
    )
    parser.add_argument(
        "--learning_rate",
        type = float,
        help = "The learning rate for training model",
        default = config.learning_rate,
    )
    parser.add_argument(
        "--max_epochs",
        type = int,
        help = "Number of epochs to train for.",
        default = config.max_epochs,
    )
    parser.add_argument(
        "--raw_dir", type = str, default = config.raw_dir, help = "Path to raw directory"
    )
    parser.add_argument(
        "-p", "--processed_dir", type = str, default = config.processed_dir, help = "."
    )

    parser.add_argument(
        "--in_channels",
        type = int,
        default = config.in_channels,
        help = "Number of input channels",
    )
    parser.add_argument(
        "--out_channels",
        type = int,
        default = config.out_channels,
        help = "Number of output channels",
    )
    parser.add_argument(
        "--depth",
        type = int,
        help = "Depth of the encoders (CNN only)",
        default = config.depth,
    )
    parser.add_argument(
        "--n_encoders",
        type = int,
        help = "Number of encoders (Unet only)",
        default = config.n_encoders,
    )
    parser.add_argument(
        "--embedding_size",
        type = int,
        help = "Embedding size of the neural network (CNN/Unet)",
        default = config.embedding_size,
    )
    parser.add_argument(
        "--pool_sizes",
        help = "A comma separated list of pool_sizes (CNN only)",
        type = str,
        default = config.pool_sizes,
    )
    parser.add_argument(
        "--kernel_size",
        help = "Kernel size of the convolutions",
        type = int,
        default = config.kernel_size,
    )

    parse_args = parser.parse_args()

    config = ESDConfig(**parse_args.__dict__)
    train(config)