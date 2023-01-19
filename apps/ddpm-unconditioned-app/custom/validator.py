from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.data import DataLoader, Dataset
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from torch.cuda.amp import autocast

from flip import FLIP
from autoencoderkl import AutoencoderKL
from simple_network import SimpleNetwork
from ddpm import DDPMScheduler


class FLIP_VALIDATOR(Executor):
    def __init__(self, validate_task_name=AppConstants.TASK_VALIDATION, project_id="", query=""):
        super().__init__()

        self._validate_task_name = validate_task_name

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

        self.val_transforms = transforms.Compose(
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

    def get_datalist(self, dataframe, val_split=0.2):
        """ Returns datalist for validation. """
        _, val_dataframe = np.split(dataframe, [int((1 - val_split) * len(dataframe))])

        datalist = []
        for accession_id in val_dataframe["accession_id"]:
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

            except Exception as e:
                print(e)

        print(f"Found {len(datalist)} files in the validation set")
        return datalist

    @torch.no_grad()
    def local_validation(self, weights, abort_signal):
        self.model.load_state_dict(state_dict=weights)
        self.model.eval()
        self.autoencoder.eval()

        epoch_loss = 0
        for step, batch in enumerate(self._test_loader):

            if abort_signal.triggered:
                # If abort_signal is triggered, we simply return.
                # The outside function will check it again and decide steps to take.
                return

            images = batch["image"].to(self.device)

            with autocast(enabled=True):
                latents = self.autoencoder.encode_stage_2_inputs(images)

                noise = torch.randn_like(latents).to(self.device)
                timesteps = torch.randint(
                    0, self.scheduler.num_train_timesteps, (images.shape[0],), device=self.device
                ).long()
                noisy_image = self.scheduler.add_noise(original_samples=latents, noise=noise, timesteps=timesteps)
                prediction = self.model.diffusion(x=noisy_image, timesteps=timesteps)

                loss = F.mse_loss(prediction.float(), noise.float())

            epoch_loss += loss.item()

        return epoch_loss / (step + 1)

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        model_owner = "?"
        try:
            if task_name == self._validate_task_name:
                test_dict = self.get_datalist(self.dataframe)
                self._test_dataset = Dataset(test_dict, transform=self.val_transforms)
                self._test_loader = DataLoader(self._test_dataset, batch_size=2, shuffle=False)

                # Get model weights
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Error in extracting dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data_kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_exception(
                        fl_ctx,
                        f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.",
                    )
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Extract weights and ensure they are tensor.
                model_owner = shareable.get_header(AppConstants.MODEL_OWNER, "?")
                weights = {k: torch.as_tensor(v, device=self.device) for k, v in dxo.data.items()}

                validation_loss = self.local_validation(weights, abort_signal)
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                dxo = DXO(data_kind=DataKind.METRICS, data={"validation_loss": validation_loss})
                return dxo.to_shareable()

            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)

        except Exception as e:
            self.log_exception(fl_ctx, f"Exception in validating model from {model_owner}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)