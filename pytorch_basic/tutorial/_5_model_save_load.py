import torch


def model_save_load(model, mode, model_dir):
    if mode == "save":
        torch.save(model.state_dict(), model_dir)
        print(f"weights are saved successfuly!")
    if mode == "load":
        # load model if exists
        try:
            model.load_state_dict(torch.load(model_dir))
        except:
            print(
                f"no weights are loaded as either {model_dir} cannot be found or incompatible to current model.")
        else:
            print(f"weights are loaded successfuly!")
