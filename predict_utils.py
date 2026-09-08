# predict_utils.py
# =============================================================================
# Shared prediction-unpacking helper (Phase-3 hardening, Reviewer #22).
#
# Modern pytorch-forecasting (>= 0.9) returns a 5-field `Prediction` namedtuple
# from model.predict(..., return_index=True):
#       Prediction(output, x, index, decoder_lengths, y)
# while older versions returned a plain 2-tuple (output, index). Every call
# site in the repository (ablation_runner, production_engine, deployment) must
# unpack through ONE canonical helper so no path crashes on the version
# mismatch documented elsewhere in this repo.
# =============================================================================

def unpack_predictions(result):
    """
    Converts the return value of
        model.predict(dataloader, mode="quantiles", return_index=True)
    into (pred_values_numpy, index_df).

    - PTF >= 0.9: result is a Prediction namedtuple with .output (tensor) and
      .index (DataFrame).
    - PTF < 0.9: result is a (output_tensor, index_df) tuple.

    pred_values_numpy shape: (n_samples, prediction_length, n_quantiles).
    """
    if hasattr(result, "output") and hasattr(result, "index"):
        pred_values = result.output.cpu().numpy()
        index_df = result.index.copy()
    elif isinstance(result, (tuple, list)) and len(result) >= 2:
        pred_values = result[0].cpu().numpy()
        index_df = result[1].copy()
    else:
        raise TypeError(
            f"Unexpected predict() return type {type(result)}. "
            "Expected a PTF Prediction namedtuple or a (output, index) tuple."
        )
    return pred_values, index_df
