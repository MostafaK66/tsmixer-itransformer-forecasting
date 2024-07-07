import os

import matplotlib.pyplot as plt

import settings


class Plotting:
    def __init__(self):
        self.output_dir = "outputs"
        self.data_name = settings.DATA_NAME
        self.output_name_tsmixer_itransformer = (
            settings.OUTPUT_NAME_TSMIXER_iTRANSFORMER
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_Y_df(self, Y_df, filename=f"Y_df_for_{settings.DATA_NAME}.png"):
        plt.figure(figsize=(10, 6))
        plt.plot(Y_df["ds"], Y_df["y"], label="Y_df")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title("Y_df Plot")
        plt.legend()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved to {output_path}")

    def plot_actuals_vs_predictions(
        self, actuals, predictions_tsmixer, predictions_itransformer, filename=None
    ):
        if filename is None:
            filename = f"actuals_vs_predictions_for_{self.data_name}.png"
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

        axes[0].plot(actuals.index, actuals, label="Actuals")
        axes[0].plot(
            predictions_tsmixer.index,
            predictions_tsmixer,
            label="TSMixer Predictions",
            linestyle="--",
        )
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Value")
        axes[0].set_title("Actuals vs TSMixer Predictions")
        axes[0].legend()

        axes[1].plot(actuals.index, actuals, label="Actuals")
        axes[1].plot(
            predictions_itransformer.index,
            predictions_itransformer,
            label="iTransformer Predictions",
            linestyle="--",
        )
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Value")
        axes[1].set_title("Actuals vs iTransformer Predictions")
        axes[1].legend()

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved to {output_path}")
