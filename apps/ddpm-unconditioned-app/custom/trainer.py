import os.path
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.data import DataLoader, Dataset
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import make_model_learnable, model_learnable_to_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.pt.pt_fed_utils import PTModelPersistenceFormatManager
from torch.cuda.amp import GradScaler, autocast

from flip import FLIP
from pt_constants import PTConstants
from autoencoderkl import AutoencoderKL
from simple_network import SimpleNetwork
from ddpm import DDPMScheduler


class FLIP_TRAINER(Executor):
    def __init__(
        self,
        lr=0.01,
        epochs=5,
        train_task_name=AppConstants.TASK_TRAIN,
        submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
        exclude_vars=None,
        project_id="",
        query="",
    ):
        super().__init__()

        self._lr = lr
        self._epochs = epochs
        self._train_task_name = train_task_name
        self._submit_model_task_name = submit_model_task_name
        self._exclude_vars = exclude_vars

        self.model = SimpleNetwork()
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            beta_start=0.0015,
            beta_end=0.0195,
        )

        self.autoencoder = AutoencoderKL(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_channels=32,
            latent_channels=3,
            ch_mult=(1, 2, 2),
            num_res_blocks=1,
            norm_num_groups=16,
            attention_levels=(False, False, True),
        )

        working_dir = Path(__file__).parent.resolve()
        model_path = working_dir / "autoencoderkl.pt"

        if not model_path.exists():
            print(f"File does not exist: {str(model_path)}")
            return

        self.autoencoder.load_state_dict(torch.load(str(model_path)))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.autoencoder.to(self.device)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-4)

        self.scaler = GradScaler()

        # Setup transforms using dictionary-based transforms.
        self._train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"], reader="NiBabelReader", as_closest_canonical=False),
                transforms.EnsureChannelFirstd(keys=["image"]),
                transforms.ScaleIntensityRanged(keys=["image"], a_min=-15, a_max=100, b_min=0, b_max=1, clip=True),
                transforms.CenterSpatialCropd(keys=["image"], roi_size=(512, 512, 256)),
                transforms.SpatialPadd(keys=["image"], spatial_size=(512, 512, 256)),
                transforms.Resized(keys=["image"], spatial_size=(96, 96, 48)),
                transforms.ToTensord(keys=["image"]),
            ]
        )

        # Setup the training dataset
        self.flip = FLIP()
        self.project_id = project_id
        self.query = query
        self.dataframe = self.flip.get_dataframe(self.project_id, self.query)

        # Setup the persistence manager to save PT model.
        # The default training configuration is used by persistence manager in case no initial model is found.
        self._default_train_conf = {"train": {"model": type(self.model).__name__}}
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=self.model.state_dict(), default_train_conf=self._default_train_conf
        )

        self.project_id = project_id
        self.query = query

    def get_datalist(self, dataframe, val_split=0.2):
        """Returns datalist for training."""
        train_dataframe, _ = np.split(dataframe, [int((1 - val_split) * len(dataframe))])

        datalist = []
        for accession_id in train_dataframe["accession_id"]:
            try:
                image_data_folder_path = self.flip.get_by_accession_number(self.project_id, accession_id)
                # TODO: Not working in testing docker container
                accession_folder_path = Path(image_data_folder_path) / accession_id
                # accession_folder_path = Path(image_data_folder_path)

                for image in list(accession_folder_path.rglob("*.nii*")):
                    header = nib.load(str(image))

                    # check is 3D and at least 128x128x128 in size
                    if len(header.shape) == 3 and all([dim >= 128 for dim in header.shape]):
                        datalist.append({"image": str(image)})
            except:
                pass

        print(f"Found {len(datalist)} files in the training set")
        return datalist

    def local_train(self, fl_ctx, weights, abort_signal):
        self.model.load_state_dict(state_dict=weights)
        self.model.train()
        self.autoencoder.eval()

        for epoch in range(self._epochs):
            epoch_loss = 0
            for step, batch in enumerate(self._train_loader):

                if abort_signal.triggered:
                    # If abort_signal is triggered, we simply return.
                    # The outside function will check it again and decide steps to take.
                    return

                images = batch["image"].to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=True):
                    with torch.no_grad():
                        latents = self.autoencoder.encode_stage_2_inputs(images)

                    noise = torch.randn_like(latents).to(self.device)
                    timesteps = torch.randint(
                        0, self.scheduler.num_train_timesteps, (images.shape[0],), device=self.device
                    ).long()

                    noisy_image = self.scheduler.add_noise(original_samples=latents, noise=noise, timesteps=timesteps)
                    prediction = self.model.diffusion(x=noisy_image, timesteps=timesteps)

                    loss = F.mse_loss(prediction.float(), noise.float())

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()

            self.log_info(fl_ctx, f"Epoch: {epoch+1}/{self._epochs} Loss: {epoch_loss / (step + 1)}")
            self.flip.send_metrics_value("Loss", epoch_loss / (step + 1), fl_ctx)

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:

        try:
            if task_name == self._train_task_name:
                train_dict = self.get_datalist(self.dataframe)
                self._train_dataset = Dataset(train_dict, transform=self._train_transforms)
                self._train_loader = DataLoader(self._train_dataset, batch_size=2, shuffle=True, num_workers=1)
                self._n_iterations = len(self._train_loader)

                # Get model weights
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Unable to extract dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_error(
                        fl_ctx,
                        f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.",
                    )
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                torch_weights = {k: torch.as_tensor(v) for k, v in dxo.data.items()}
                self.local_train(fl_ctx, torch_weights, abort_signal)

                # Check the abort_signal after training.
                # local_train returns early if abort_signal is triggered.
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                self.save_local_model(fl_ctx)

                # Get the new state dict and send as weights
                new_weights = self.model.state_dict()
                new_weights = {k: v.cpu().numpy() for k, v in new_weights.items()}

                outgoing_dxo = DXO(
                    data_kind=DataKind.WEIGHTS,
                    data=new_weights,
                    meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations},
                )
                return outgoing_dxo.to_shareable()

            elif task_name == self._submit_model_task_name:
                ml = self.load_local_model(fl_ctx)
                dxo = model_learnable_to_dxo(ml)
                return dxo.to_shareable()

            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)

        except:
            self.log_exception(fl_ctx, f"Exception in simple trainer.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def save_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        ml = make_model_learnable(self.model.state_dict(), {})
        self.persistence_manager.update(ml)
        torch.save(self.persistence_manager.to_persistence_dict(), model_path)

    def load_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            return None
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        self.persistence_manager = PTModelPersistenceFormatManager(
            data=torch.load(model_path), default_train_conf=self._default_train_conf
        )
        ml = self.persistence_manager.to_model_learnable(exclude_vars=self._exclude_vars)
        return ml
