import os
import time

os.environ["OMP_NUM_THREADS"] = "1"
import settings
from data_loader import DataLoader
from multivariate_predictor import MultivariatePredictor
from plotting import Plotting
from univariate_predictor import UnivariatePredictor


def main():
    start_time = time.time()
    data_loader = DataLoader(settings.DATA_NAME)
    plotter = Plotting()
    Y_df = data_loader.data
    val_size = data_loader.val_size
    test_size = data_loader.test_size
    freq = data_loader.freq

    plotter.plot_Y_df(Y_df)

    if settings.DATA_NAME == "ettm1":
        predictor = UnivariatePredictor()
        (
            actuals,
            predictions_tsmixer,
            predictions_itransformer,
        ) = predictor.run_prediction(
            Y_df=Y_df, val_size=val_size, test_size=test_size, freq=freq
        )
    elif settings.DATA_NAME == "ettm2":
        predictor = MultivariatePredictor()
        (
            actuals,
            predictions_tsmixer,
            predictions_itransformer,
        ) = predictor.run_prediction(
            Y_df=Y_df, val_size=val_size, test_size=test_size, freq=freq
        )
    else:
        raise ValueError(f"Unsupported DATA_NAME: {settings.DATA_NAME}")

    plotter.plot_actuals_vs_predictions(
        actuals, predictions_tsmixer, predictions_itransformer
    )

    end_time = time.time()
    total_time = (end_time - start_time) / 60
    print(f"Total running timez: {total_time:.2f} minutes")


if __name__ == "__main__":
    main()
