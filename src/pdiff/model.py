"""
Copyright (c) 2026 Novartis Biomedical Research Inc. Licensed under the MIT License. See LICENSE file in the project root.
"""

from typing import Any, Dict
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    SchedulerMixin,
    StableDiffusionPipeline,
)
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
import torch
from pathlib import Path
from typing_extensions import Self

from pdiff.metadata import pDiffMetadata


class pDiffModel:
    def __init__(
        self,
        model_path: Path = None,
        pipeline: StableDiffusionPipeline = None,
        profile_length: int = 2048,
        from_scratch: bool = False,
    ):
        """initialize a pDiffModel from an existing pipeline or load a model from disk. Set the profile length as desired in the latter case

        Args:
            model_path (Path, optional): path to load a pipeline from disk. If it is set, pipeline cannot be set. Defaults to None.
            pipeline (StableDiffusionPipeline, optional): Existing pipeline to encapsulate. If it is set, model_path cannot be set. Defaults to None.
            profile_length (int, optional): if loading a pipeline from a model path, will reset the cross_attention_dim to this. Defaults to 2048.
            from_scratch (bool, optional): if loading a pipeline from a model path, will reinitialize the model weights. Defaults to False.
        """
        self.model_path = None
        assert not ((pipeline is None) and (model_path is None))
        assert not ((pipeline is not None) and (model_path is not None))
        if pipeline is not None:
            self.set_pipeline(pipeline)
        if model_path is not None:
            self.load(model_path, profile_length, from_scratch)

    @staticmethod
    def piecewise_initialize_pipeline(
        pipeline_root_path: Path, unet: UNet2DConditionModel
    ) -> StableDiffusionPipeline:
        """loads all stable diffusion pipeline components from pretrained except for the unet. Useful when
        you don't want to init the unet from a config alone to save space but the VAE, etc. are pretrained

        Args:
            pipeline_root_path (Path): path containing folders scheduler, text_encoder, tokenizer, vae
            unet (UNet2DConditionModel): the unet to wrap in the final pipeline

        Returns:
            StableDiffusionPipeline: initialized stable diffusion pipeline
        """
        vae = AutoencoderKL.from_pretrained(pipeline_root_path / "vae")
        text_encoder = CLIPTextModel.from_pretrained(
            pipeline_root_path / "text_encoder"
        )
        tokenizer = CLIPTokenizer.from_pretrained(pipeline_root_path / "tokenizer")
        scheduler = DDIMScheduler.from_pretrained(pipeline_root_path / "scheduler")
        pipeline = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        return pipeline

    def load(
        self, load_path: Path, profile_length: int = 2048, from_scratch: bool = False
    ) -> Self:
        if from_scratch:
            unet_config = UNet2DConditionModel.load_config(load_path / "unet")
            unet_config["cross_attention_dim"] = profile_length
            unet = UNet2DConditionModel.from_config(unet_config)
        else:
            unet = UNet2DConditionModel.from_pretrained(
                load_path,
                subfolder="unet",
                cross_attention_dim=profile_length,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
            )
        pipeline = self.piecewise_initialize_pipeline(load_path, unet)
        # self.pipeline = StableDiffusionPipeline.from_pretrained(load_path)
        self.pipeline = pipeline
        # self.set_unet(unet)
        self.pipeline.vae.requires_grad_(False)
        self.model_path = load_path
        return self

    def save(self, save_path: Path) -> None:
        if save_path == self.model_path:
            print("saving in same folder as source, skipping")
            return
        self.pipeline.save_pretrained(save_path)
        self.model_path = save_path

    def get_unet(self) -> UNet2DConditionModel:
        return self.pipeline.unet

    def set_unet(self, unet: UNet2DConditionModel) -> None:
        self.pipeline.unet = unet

    def set_scheduler(self, scheduler: SchedulerMixin) -> None:
        self.pipeline.scheduler = scheduler

    def get_scheduler(self) -> SchedulerMixin:
        return self.pipeline.scheduler

    def set_vae(self, vae: AutoencoderKL) -> None:
        self.pipeline.vae = vae

    def get_vae(self) -> AutoencoderKL:
        return self.pipeline.vae

    def set_pipeline(self, pipeline: StableDiffusionPipeline) -> None:
        pipeline.vae.requires_grad_(False)
        self.pipeline = pipeline

    def get_pipeline(self) -> StableDiffusionPipeline:
        return self.pipeline

    def to_device(self, device: torch.device) -> Self:
        self.pipeline = self.pipeline.to(device)
        return self

    def predict(
        self,
        output_root_path: Path,
        new_metadata_filename: str,
        treatment_profile_dict: Dict[Any, np.ndarray],
        gen_images_per_treatment: int = 1,
        inference_steps: int = 100,
        guidance_scale=0,
        resolution: int = 512,
        random_seed: int = 0,
    ) -> pDiffMetadata:
        sample_prompt = next(iter(treatment_profile_dict.values()))
        negative_prompt = torch.zeros_like(
            torch.Tensor(sample_prompt.reshape(1, 1, -1))
        )
        output_root_path.mkdir(exist_ok=True, parents=True)
        prediction_pdiff_metadata_path = output_root_path / new_metadata_filename
        pDiffMetadata.initialize_dataframe(prediction_pdiff_metadata_path)
        prediction_pdiff_metadata = pDiffMetadata(prediction_pdiff_metadata_path)
        with torch.autocast("cuda"):
            generator = [
                # torch.Generator("cuda").manual_seed(i + random_seed)
                torch.Generator(self.pipeline.device).manual_seed(i + random_seed)
                for i in range(gen_images_per_treatment)
            ]
            for treatment in sorted(treatment_profile_dict.keys()):
                prompt = torch.Tensor(
                    treatment_profile_dict[treatment].reshape(1, 1, -1)
                )
                inference_result = self.pipeline(
                    prompt_embeds=prompt,
                    negative_prompt_embeds=negative_prompt,
                    num_images_per_prompt=gen_images_per_treatment,
                    guidance_scale=guidance_scale,
                    height=resolution,
                    width=resolution,
                    num_inference_steps=inference_steps,
                    generator=generator,
                )
                prediction_pdiff_metadata.add_image_data(
                    output_root_path,
                    inference_result["images"],
                    treatment,
                    treatment_profile_dict[treatment],
                )
                torch.cuda.empty_cache()
        return prediction_pdiff_metadata
