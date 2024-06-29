from pathlib import Path

def get_weights_file_path(epoch: str):
    model_folder = "weights"
    model_basename = "tmodel_"
    model_filename = f"{model_basename}{epoch}.pt"
    # return str(Path('../../../Me/ERA/S17/S16_code') / model_folder / model_filename)
    return str(Path('.') / model_folder / model_filename)


