"""
Copyright (c) 2026 Novartis Biomedical Research Inc. Licensed under the MIT License. See LICENSE file in the project root.
"""

import torch
from pdiff.dataset import pDiffDataset
from pdiff.model import pDiffModel
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
from diffusers import StableDiffusionPipeline, UNet2DConditionModel


def collate_fn_pdiffdataset(examples):
    image = torch.stack([example["image"] for example in examples])
    image = image.to(memory_format=torch.contiguous_format).float()
    profile = torch.stack([example["profile"] for example in examples]).float()
    return {"image": image, "profile": profile}


def test_init_from_pipeline(pdiff_model: pDiffModel):
    model_path = Path(__file__).resolve().parent / "test_data/sample_pdiff_model"
    unet_config = UNet2DConditionModel.load_config(model_path / "unet")
    # unet_config["cross_attention_dim"] = profile_length
    unet = UNet2DConditionModel.from_config(unet_config)
    pipeline = pDiffModel.piecewise_initialize_pipeline(model_path, unet)
    # pipeline = StableDiffusionPipeline.from_pretrained(model_path)
    pipeline_init_pdiff_model = pDiffModel(pipeline=pipeline)
    vae_1 = pipeline_init_pdiff_model.get_vae()
    vae_2 = pdiff_model.get_vae()
    assert vae_1.num_parameters() == vae_2.num_parameters()


def test_get_vae(pdiff_model: pDiffModel):
    vae = pdiff_model.get_vae()
    assert vae.num_parameters() == 83653863


def test_get_unet(pdiff_model: pDiffModel):
    unet = pdiff_model.get_unet()
    assert unet.num_parameters() == 891469764


def test_load_save(pdiff_model: pDiffModel, tmp_path):
    save_path = tmp_path / "save"
    pdiff_model.save(save_path)
    loaded_model = pDiffModel(model_path=save_path)
    assert loaded_model.get_unet().num_parameters() == 891469764


def test_load_from_scratch(tmp_path):
    model_path = Path(__file__).resolve().parent / "test_data/sample_pdiff_model"
    scratch_model = pDiffModel(model_path=model_path, from_scratch=True)
    assert scratch_model.get_unet().num_parameters() == 891469764


def test_dataset_with_unet(pdiff_model: pDiffModel, pdiff_dataset: pDiffDataset):
    with torch.no_grad():
        bsz = 1
        my_dataloader = DataLoader(
            pdiff_dataset, batch_size=bsz, collate_fn=collate_fn_pdiffdataset
        )
        batch = next(iter(my_dataloader))
        vae = pdiff_model.get_vae()
        unet = pdiff_model.get_unet()
        noise_scheduler = pdiff_model.get_scheduler()
        unet.eval()
        latents = vae.encode(batch["image"].to(torch.float)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        assert latents.ndim == 4
        assert latents.shape == (1, 4, 64, 64)
        encoder_hidden_states = batch["profile"]  # .to(torch.float)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()
        model_pred = unet(latents, timesteps, encoder_hidden_states).sample
        assert model_pred.shape == latents.shape
