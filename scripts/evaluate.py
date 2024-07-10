import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.append(".")
from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.utilities import ESDConfig
from src.visualization.restitch_plot import restitch_and_plot

def list_validation_tiles(processed_dir):
    processed_dir = Path(processed_dir)
    val_tiles = list((processed_dir / "Val" / "subtiles").glob("*"))
    return val_tiles
def main(options):
    # initialize datamodule
    datamodule = ESDDataModule(options.processed_dir,
                                options.raw_dir,
                                options.batch_size,
                                options.num_workers,
                                options.seed,
                                options.selected_bands,
                                options.slice_size)
    # prepare data
    datamodule.prepare_data()
    datamodule.setup(stage = 'fit')
    # load model from checkpoint
    # set model to eval mode
    model = ESDSegmentation.load_from_checkpoint(options.model_path)

    model.eval()
    # get a list of all processed tiles
    processed_tiles = list_validation_tiles(options.processed_dir)
    # for each tile
    for tile_dir in processed_tiles:
        # run restitch and plot
        restitch_and_plot(
            options=options,
            datamodule=datamodule,
            model=model,
            results_dir=Path(options.results_dir),
            parent_tile_id = tile_dir.name,
            accelerator = 'cpu'  #'cpu', adjust to 'gpu' if using gpu
        )


if __name__ == "__main__":
    config = ESDConfig()
    parser = ArgumentParser()

    parser.add_argument(
        "--model_path", type=str, help="Model path.", default=config.model_path
    )
    parser.add_argument(
        "--raw_dir", type=str, default=config.raw_dir, help="Path to raw directory"
    )
    parser.add_argument(
        "-p", "--processed_dir", type=str, default=config.processed_dir, help="."
    )
    parser.add_argument(
        "--results_dir", type=str, default=config.results_dir, help="Results dir"
    )
    main(ESDConfig(**parser.parse_args().__dict__))