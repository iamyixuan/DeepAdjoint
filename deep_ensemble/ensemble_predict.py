import numpy as np
import pandas as pd
import torch
from data import SOMAdata
from metrics import MSE_torch, log_likelihood_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_model_nll import FNO_MU_STD as FNO_nll
from train_model_quantile import FNO_QR as FNO_quantile

"""
This script is to form the ensemble and make predicitons on the testing set
There are few functions needed:
1. load the search results and return the list of ensembles based on either
top K or Caruana's method.
2. load the members and make predictions on the testing set,
save all the predictions
and calcuate the ensemble predictions.
3. with the ensemble predictions, calculate the accuracy and other metrics.
4. enable rollout predictions as well.
5. plot the uncertanties and metrics and rollout metrics.
"""


class EnsemblePredictor:
    def __init__(
        self, hpo_results_path, model_dir, criterion, val_data, test_data
    ):
        self.hpo_results = self.clean_df(pd.read_csv(hpo_results_path))
        self.val_loader = DataLoader(
            val_data, batch_size=val_data.__len__(), shuffle=False
        )
        self.test_loader = DataLoader(
            test_data, batch_size=test_data.__len__(), shuffle=False
        )
        self.model_dir = model_dir
        self.criterion = criterion
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def clean_df(self, df):
        df = df[df["objective_0"] != "F"]
        df.loc[:, "objective_0"] = df["objective_0"].values.astype(float)
        df.loc[:, "objective_1"] = df["objective_1"].values.astype(float)
        df = df.dropna()
        df = df.rename(columns=lambda x: x.replace("p:", ""))
        return df

    def extract_configs(self, job_id):
        df = self.hpo_results.copy()
        index = df[df["job_id"] == job_id].index[0]
        return df.loc[index]

    def select_topK(self, K):
        df = self.hpo_results.copy()
        df["objective_sum"] = df["objective_0"] + df["objective_1"]
        sorted_df = df.sort_values(by="objective_sum", ascending=False)
        return sorted_df["job_id"][:K].values

    def select_caruana(self, K):
        # start with the best model
        df = self.hpo_results.copy()
        df["objective_sum"] = df["objective_0"] + df["objective_1"]
        sorted_df = df.sort_values(by="objective_sum", ascending=False)
        model_ids = [sorted_df.iloc[0]["job_id"]]
        available_ids = list(sorted_df["job_id"].values)[:100]  # top 100
        while len(model_ids) < K:
            print(f"Selected {len(model_ids)}/{K} models")
            best_id = model_ids[-1]
            best_score = -np.inf
            for i in tqdm(available_ids):
                model_ids.append(i)
                ensemble_predictions, true_labels = self.ensemble_predict_val(
                    model_ids
                )
                pred_mean, aleatoric_uc, epistemic_uc = (
                    self.forecast_and_unceartainties(
                        ensemble_predictions, true_labels
                    )
                )

                score = log_likelihood_score(
                    loc=pred_mean,
                    scale=torch.sqrt(aleatoric_uc),
                    sample=true_labels,
                ) - MSE_torch(pred_mean, true_labels)

                if score > best_score:
                    best_score = score
                    best_id = i
                model_ids.pop()
            model_ids.append(best_id)
            available_ids.remove(best_id)
        return model_ids

    def ensemble_predict(self, models, data_loader):
        ensemble_predictions = []
        assert len(models) > 0

        for model in models:
            model.eval()
            model.to(self.device)
            with torch.no_grad():
                for data in data_loader:
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(inputs)
                    ensemble_predictions.append(outputs)

        ensemble_predictions = torch.stack(ensemble_predictions)
        return ensemble_predictions, labels

    def _ensemble_predict(self, models, x, y):
        ensemble_predictions = []
        for model in models:
            model.eval()
            model.to(self.device)
            with torch.no_grad():
                outputs = model(x)
                ensemble_predictions.append(outputs)
        ensemble_predictions = torch.stack(ensemble_predictions)
        return ensemble_predictions, y

    def ensemble_predict_val(self, model_ids):
        """
        Only use for Caruana's method
        """
        assert self.criterion == "caruana"
        models = self.load_models(model_ids, self.model_dir)

        ensemble_predictions, true_labels = self.ensemble_predict(
            models, self.val_loader
        )
        return ensemble_predictions, true_labels

    def predict_test(self):
        if self.criterion == "topK":
            model_ids = self.select_topK(10)  # select top 10
        elif self.criterion == "caruana":
            model_ids = self.select_caruana(10)

        models = self.load_models(model_ids, self.model_dir)
        ensemble_predictions, true_labels = self.ensemble_predict(
            models, self.test_loader
        )
        return ensemble_predictions, true_labels

    def load_models(self, model_ids, model_dir):
        model_list = []
        for model_id in model_ids:
            config = self.extract_configs(model_id)
            if config["loss"] == "nll":
                model = FNO_nll(
                    in_channels=5,
                    out_channels=4 * 2,
                    decoder_layers=config["num_projs"],
                    decoder_layer_size=config["proj_size"],
                    decoder_activation_fn=config["proj_act"],
                    dimension=2,
                    latent_channels=config["latent_ch"],
                    num_fno_layers=config["num_FNO"],
                    num_fno_modes=int(config["num_modes"]),
                    padding=config["padding"],
                    padding_type=config["padding_type"],
                    activation_fn=config["lift_act"],
                    coord_features=config["coord_feat"],
                )
            elif config["loss"] == "quantile":
                model = FNO_quantile(
                    in_channels=5,
                    out_channels=4 * 3,
                    decoder_layers=config["num_projs"],
                    decoder_layer_size=config["proj_size"],
                    decoder_activation_fn=config["proj_act"],
                    dimension=2,
                    latent_channels=config["latent_ch"],
                    num_fno_layers=config["num_FNO"],
                    num_fno_modes=int(config["num_modes"]),
                    padding=config["padding"],
                    padding_type=config["padding_type"],
                    activation_fn=config["lift_act"],
                    coord_features=config["coord_feat"],
                )

            model.load_state_dict(torch.load(f"{model_dir}/0.{model_id}.pth"))
            model_list.append(model)
        return model_list

    def forecast_and_unceartainties(self, ensemble_predictions, true_labels):
        assert len(ensemble_predictions.shape) == 5
        print(true_labels.shape)
        ch = true_labels.shape[1]
        ensemble_mean = torch.mean(ensemble_predictions, dim=0)

        if ensemble_predictions.shape[2] == 2 * ch:
            pred_mean = ensemble_mean[:, :ch]
            pred_std = ensemble_mean[:, ch:]
            aleatoric_uc = pred_std**2
            epistemic_uc = torch.var(ensemble_predictions, dim=0)[:, :ch]
        elif ensemble_predictions.shape[2] == 3 * ch:
            pred_mean = ensemble_mean[:, ch : 2 * ch]
            pred_std = 0.5 * torch.abs(
                ensemble_mean[:, 2 * ch :] - ensemble_mean[:, :ch]
            )
            aleatoric_uc = pred_std**2
            epistemic_uc = torch.var(ensemble_predictions, dim=0)[
                :, ch : 2 * ch
            ]
        else:
            raise ValueError("The ensemble predictions are not correct")
        return pred_mean, aleatoric_uc, epistemic_uc

    def map_to_physical_scale(self, x, if_var=False):
        """
        x: (B, 4, 100, 100)
        """
        assert len(x.shape) == 4
        datamin = np.array([34.01481, 5.144762, 3.82e-8, 6.95e-9])
        datamax = np.array([34.24358, 18.84177, 0.906503, 1.640676])
        datamin = datamin[None, :, None, None]
        datamax = datamax[None, :, None, None]
        if if_var:
            x = x * (datamax - datamin) ** 2
        else:
            x = x * (datamax - datamin) + datamin
        return x

    def ensemble_rollout(self, dataloader):
        if self.criterion == "topK":
            model_ids = self.select_topK(10)
        elif self.criterion == "caruana":
            model_ids = self.select_caruana(10)
        models = self.load_models(model_ids, self.model_dir)

        # for every 29 steps we save a sequence
        truePred = {"true": [], "pred": [], "al_uc": [], "ep_uc": []}
        temp_pred = []
        temp_al = []
        temp_ep = []
        true = []
        for i, (x, y) in enumerate(dataloader):
            assert x.shape[0] == 1
            x = x.to(self.device)
            y = y.to(self.device)
            if i % 29 == 0 and i > 0:
                truePred["true"].append(true)
                truePred["pred"].append(temp_pred)
                truePred["al_uc"].append(temp_al)
                truePred["ep_uc"].append(temp_ep)

                ensemble_predictions, true_labels = self._ensemble_predict(
                    models, x, y
                )
                pred, al_uc, ep_uc = self.forecast_and_unceartainties(
                    ensemble_predictions, true_labels
                )
                temp_pred = [pred]  # reset sequence
                temp_al = [al_uc]
                temp_ep = [ep_uc]
                true = [y]
            else:
                if len(temp_pred) != 0:
                    x_cat = torch.cat([temp_pred[-1], x[:, -1:]], dim=1)
                    ensemble_predictions, true_labels = self._ensemble_predict(
                        models, x_cat, y
                    )
                    pred, al_uc, ep_uc = self.forecast_and_unceartainties(
                        ensemble_predictions, true_labels
                    )
                    temp_pred.append(pred)
                    temp_al.append(al_uc)
                    temp_ep.append(ep_uc)
                    true.append(y)
                else:
                    ensemble_predictions, true_labels = self._ensemble_predict(
                        models, x, y
                    )
                    pred, al_uc, ep_uc = self.forecast_and_unceartainties(
                        ensemble_predictions, true_labels
                    )
                    temp_pred = [pred]
                    temp_al = [al_uc]
                    temp_ep = [ep_uc]
                    true = [y]
        return truePred


if __name__ == "__main__":

    valset = SOMAdata(
        "/pscratch/sd/y/yixuans/datatset/de_dataset/GM-prog-var-surface.hdf5",
        "val",
        y_noise=True,
        transform=True,
    )
    testSet = SOMAdata(
        "/pscratch/sd/y/yixuans/datatset/de_dataset/GM-prog-var-surface.hdf5",
        "test",
        transform=True,
    )
    predictor = EnsemblePredictor(
        hpo_results_path="./hpo_logs/hpo_nll_quantile_checkpointing/results.csv",
        model_dir="/pscratch/sd/y/yixuans/saved_models_hpo/",
        criterion="topK",
        val_data=valset,
        test_data=testSet,
    )

    # make predictions with top k
    ensemble_predictions, true_labels = predictor.predict_test()
    pred_mean, aleatoric_uc, epistemic_uc = (
        predictor.forecast_and_unceartainties(
            ensemble_predictions, true_labels
        )
    )
    true_labels = predictor.map_to_physical_scale(true_labels.cpu().numpy())
    pred_mean = predictor.map_to_physical_scale(pred_mean.cpu().numpy())
    aleatoric_uc = predictor.map_to_physical_scale(
        aleatoric_uc.cpu().numpy(), if_var=True
    )
    epistemic_uc = predictor.map_to_physical_scale(
        epistemic_uc.cpu().numpy(), if_var=True
    )

    # save the predictions
    np.savez(
        "/pscratch/sd/y/yixuans/hpo_uq_pred/topK_test_ensemble_predictions.npz",
        true_labels=true_labels,
        pred_mean=pred_mean,
        aleatoric_uc=aleatoric_uc,
        epistemic_uc=epistemic_uc,
    )

    # generate rollout
    test_rollout_loader = DataLoader(testSet, batch_size=1, shuffle=False)
    rollouts = predictor.ensemble_rollout(test_rollout_loader)

    def process_rollout(data, if_var=False):
        process_data = []
        for d in data:
            d = np.asarray([p.cpu().detach().numpy() for p in d]).squeeze()
            d = predictor.map_to_physical_scale(d, if_var)
            process_data.append(d)
        return np.asarray(process_data)

    true = process_rollout(rollouts["true"])
    pred = process_rollout(rollouts["pred"])
    al_uc = process_rollout(rollouts["al_uc"], if_var=True)
    ep_uc = process_rollout(rollouts["ep_uc"], if_var=True)

    np.savez(
        "/pscratch/sd/y/yixuans/hpo_uq_pred/topK_test_rollout.npz",
        true=true,
        pred=pred,
        al_uc=al_uc,
        ep_uc=ep_uc,
    )

    # val predict vs num of models
    model_ids = predictor.select_topK(10)
    models = predictor.load_models(model_ids, predictor.model_dir)
    pred_mean_list = []
    aleatoric_uc_list = []
    epistemic_uc_list = []

    for i in range(len(models)):
        cur_models = list(models[: i + 1])
        pred, true = predictor.ensemble_predict(
            cur_models, predictor.val_loader
        )
        pred_mean, aleatoric_uc, epistemic_uc = (
            predictor.forecast_and_unceartainties(pred, true)
        )
        pred_mean = predictor.map_to_physical_scale(pred_mean.cpu().numpy())
        aleatoric_uc = predictor.map_to_physical_scale(
            aleatoric_uc.cpu().numpy(), if_var=True
        )
        epistemic_uc = predictor.map_to_physical_scale(
            epistemic_uc.cpu().numpy(), if_var=True
        )
        true = predictor.map_to_physical_scale(true.cpu().numpy())
        pred_mean_list.append(pred_mean)
        aleatoric_uc_list.append(aleatoric_uc)
        epistemic_uc_list.append(epistemic_uc)

    np.savez(
        "/pscratch/sd/y/yixuans/hpo_uq_pred/topK_val_pred_vs_num_members.npz",
        pred_mean=np.asarray(pred_mean_list),
        aleatoric_uc=np.asarray(aleatoric_uc_list),
        epistemic_uc=np.asarray(epistemic_uc_list),
        true=true,
    )
