import os

from neuralforecast.core import NeuralForecast
from neuralforecast.models import TSMixer, iTransformer
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mse

import settings


class MultivariatePredictor:
    def __init__(
        self,
        n_block=settings.TSMIXER_N_BLOCK,
        ff_dim=settings.TSMIXER_FF_DIM,
        learning_rate=settings.TSMIXER_LEARNING_RATE,
        batch_size=settings.TSMIXER_BATCH_SIZE,
        horizon=settings.TSMIXER_HORIZON,
        max_steps=settings.TSMIXER_MAX_STEPS,
        early_stop_patience_steps=settings.TSMIXER_EARLY_STOPPING,
        hidden_size=settings.iTRANSFORMER_HIDDEN_SIZE,
        n_heads=settings.iTRANSFORMER_N_HEADS,
        scaler_type=settings.SCALER_TYPE_MULTIVARIANT,
    ):
        self.n_block = n_block
        self.ff_dim = ff_dim
        self.batch_size = (batch_size,)
        self.learning_rate = learning_rate
        self.horizon = horizon
        self.max_steps = max_steps
        self.early_stop_patience_steps = early_stop_patience_steps
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.scaler_type = scaler_type
        self.models = [
            TSMixer(
                h=horizon,
                input_size=3 * horizon,
                ff_dim=ff_dim,
                learning_rate=learning_rate,
                batch_size=batch_size,
                n_series=1,
                max_steps=max_steps,
                n_block=n_block,
                early_stop_patience_steps=early_stop_patience_steps,
                scaler_type=scaler_type,
            ),
            iTransformer(
                h=horizon,
                hidden_size=hidden_size,
                input_size=3 * horizon,
                learning_rate=learning_rate,
                batch_size=batch_size,
                n_series=1,
                max_steps=max_steps,
                early_stop_patience_steps=early_stop_patience_steps,
                n_heads=n_heads,
                scaler_type=scaler_type,
            ),
        ]

    def run_prediction(self, Y_df, val_size, test_size, freq):
        nf = NeuralForecast(models=self.models, freq=freq)
        nf_preds = nf.cross_validation(
            df=Y_df, val_size=val_size, test_size=test_size, n_windows=None
        )
        nf_preds = nf_preds.reset_index()

        evaluation_results = evaluate(
            df=nf_preds,
            metrics=[mae, mse],
            models=["TSMixer", "iTransformer"],
        )

        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "ettm1_results.csv")
        evaluation_results.to_csv(output_path, index=False, header=True)
        print(f"Results saved to {output_path}")

        actuals = nf_preds["y"]
        predictions_tsmixer = nf_preds["TSMixer"]
        predictions_itransformer = nf_preds["iTransformer"]

        return actuals, predictions_tsmixer, predictions_itransformer
