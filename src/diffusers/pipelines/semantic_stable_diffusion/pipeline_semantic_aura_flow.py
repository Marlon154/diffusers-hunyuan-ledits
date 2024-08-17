# Copyright 2024 AuraFlow Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
from typing import List, Optional, Tuple, Union

import torch
from transformers import T5Tokenizer, UMT5EncoderModel

from ...image_processor import VaeImageProcessor
from ...models import AuraFlowTransformer2DModel, AutoencoderKL
from ...models.attention_processor import AttnProcessor2_0, FusedAttnProcessor2_0, XFormersAttnProcessor
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import AuraFlowPipeline

        >>> pipe = AuraFlowPipeline.from_pretrained("fal/AuraFlow", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt).images[0]
        >>> image.save("aura_flow.png")
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class SemanticAuraFlowPipeline(DiffusionPipeline):
    r"""
    Args:
        tokenizer (`T5TokenizerFast`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. AuraFlow uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [EleutherAI/pile-t5-xl](https://huggingface.co/EleutherAI/pile-t5-xl) variant.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        transformer ([`AuraFlowTransformer2DModel`]):
            Conditional Transformer (MMDiT and DiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKL,
        transformer: AuraFlowTransformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
        )

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                    f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                    f" {negative_prompt_attention_mask.shape}."
                )

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = None,
        do_classifier_free_guidance: bool = True,
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 256,
        editing_prompt: Optional[Union[str, List[str]]] = None,
        editing_prompt_embeds: Optional[torch.FloatTensor] = None,
        editing_prompt_attention_mask: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            max_sequence_length (`int`, defaults to 256): Maximum sequence length to use for the prompt.
        """
        if device is None:
            device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        max_length = max_sequence_length
        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            text_input_ids = text_inputs["input_ids"]
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because T5 can only handle sequences up to"
                    f" {max_length} tokens: {removed_text}"
                )

            prompt_embeds = self.text_encoder(**text_inputs)[0]
            prompt_attention_mask = text_inputs["attention_mask"].unsqueeze(-1).expand(prompt_embeds.shape)
            prompt_embeds = prompt_embeds * prompt_attention_mask

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.reshape(bs_embed, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        if editing_prompt is not None:
            if isinstance(editing_prompt, str):
                editing_prompt = [editing_prompt]
            if not isinstance(editing_prompt, list):
                raise ValueError(
                    f"`editing_prompt` has to be of type `str` or `list` but is {type(editing_prompt)}"
                )
            editing_prompt_embeds = []
            editing_prompt_attention_mask = []
            for single_prompt in editing_prompt:
                max_length = prompt_embeds.shape[1]
                edit_input = self.tokenizer(
                    single_prompt,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                edit_input = {k: v.to(device) for k, v in edit_input.items()}
                single_edit_prompts_embeds = self.text_encoder(**edit_input)[0]
                single_edit_prompt_attention_mask = (
                    edit_input["attention_mask"].unsqueeze(-1).expand(single_edit_prompts_embeds.shape)
                )
                single_edit_prompts_embeds = single_edit_prompts_embeds * single_edit_prompt_attention_mask
                editing_prompt_embeds.append(single_edit_prompts_embeds)
                editing_prompt_attention_mask.append(single_edit_prompt_attention_mask)
            editing_prompt_embeds = torch.cat(editing_prompt_embeds)
            editing_prompt_attention_mask = torch.cat(editing_prompt_attention_mask)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            uncond_tokens = [negative_prompt] * batch_size if isinstance(negative_prompt, str) else negative_prompt
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            uncond_input = {k: v.to(device) for k, v in uncond_input.items()}
            negative_prompt_embeds = self.text_encoder(**uncond_input)[0]
            negative_prompt_attention_mask = (
                uncond_input["attention_mask"].unsqueeze(-1).expand(negative_prompt_embeds.shape)
            )
            negative_prompt_embeds = negative_prompt_embeds * negative_prompt_attention_mask

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            negative_prompt_attention_mask = negative_prompt_attention_mask.reshape(bs_embed, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask, editing_prompt_embeds, editing_prompt_attention_mask

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.upcast_vae
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                FusedAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 256,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        editing_prompt: Optional[Union[str, List[str]]] = None,
        editing_prompt_embeds: Optional[torch.Tensor] = None,
        reverse_editing_direction: Optional[Union[bool, List[bool]]] = False,
        edit_guidance_scale: Optional[Union[float, List[float]]] = 5,
        edit_warmup_steps: Optional[Union[int, List[int]]] = 10,
        edit_cooldown_steps: Optional[Union[int, List[int]]] = None,
        edit_threshold: Optional[Union[float, List[float]]] = 0.9,
        edit_momentum_scale: Optional[float] = 0.1,
        edit_mom_beta: Optional[float] = 0.4,
        edit_weights: Optional[List[float]] = None,
        sem_guidance: Optional[List[torch.Tensor]] = None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.transformer.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 512 by default.
            width (`int`, *optional*, defaults to self.transformer.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 512 by default.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
                `num_inference_steps` and `timesteps` must be `None`.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns: [`~pipelines.ImagePipelineOutput`] or `tuple`:
            If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is returned
            where the first element is a list with the generated images.
        """
        # 1. Check inputs. Raise error if not correct
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )

        # 2. Determine batch size.
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if editing_prompt:
            enable_edit_guidance = True
            if isinstance(editing_prompt, str):
                editing_prompt = [editing_prompt]
            enabled_editing_prompts = len(editing_prompt)
        elif editing_prompt_embeds is not None:
            enable_edit_guidance = True
            enabled_editing_prompts = editing_prompt_embeds.shape[0]
        else:
            enabled_editing_prompts = 0
            enable_edit_guidance = False

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
            editing_prompt_embeds,
            editing_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            editing_prompt=editing_prompt,
            editing_prompt_embeds=editing_prompt_embeds,
        )
        if do_classifier_free_guidance:
            if enable_edit_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds, editing_prompt_embeds], dim=0)
            else:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare timesteps

        # sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        edit_momentum = None
        # 6. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * (2 +enabled_editing_prompts)) if do_classifier_free_guidance else latents

                # aura use timestep value between 0 and 1, with t=1 as noise and t=0 as the image
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = torch.tensor([t / 1000]).expand(latent_model_input.shape[0])
                timestep = timestep.to(latents.device, dtype=latents.dtype)

                # predict noise model_output
                noise_pred = self.transformer(
                    latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_out = noise_pred.chunk(2 + enabled_editing_prompts)  # [b,4, 64, 64]
                    noise_pred_uncond, noise_pred_text = noise_pred_out[0], noise_pred_out[1]
                    noise_pred_edit_concepts = noise_pred_out[2:]

                    # default text guidance
                    noise_guidance = guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if edit_momentum is None:
                        edit_momentum = torch.zeros_like(noise_guidance)
                    if enable_edit_guidance:
                        concept_weights = torch.zeros(
                            (len(noise_pred_edit_concepts), noise_guidance.shape[0]),
                            device=device,
                            dtype=noise_guidance.dtype,
                        )
                        noise_guidance_edit = torch.zeros(
                            (len(noise_pred_edit_concepts), *noise_guidance.shape),
                            device=device,
                            dtype=noise_guidance.dtype,
                        )
                        # noise_guidance_edit = torch.zeros_like(noise_guidance)
                        warmup_inds = []
                        for c, noise_pred_edit_concept in enumerate(noise_pred_edit_concepts):

                            if isinstance(edit_guidance_scale, list):
                                edit_guidance_scale_c = edit_guidance_scale[c]
                            else:
                                edit_guidance_scale_c = edit_guidance_scale

                            if isinstance(edit_threshold, list):
                                edit_threshold_c = edit_threshold[c]
                            else:
                                edit_threshold_c = edit_threshold
                            if isinstance(reverse_editing_direction, list):
                                reverse_editing_direction_c = reverse_editing_direction[c]
                            else:
                                reverse_editing_direction_c = reverse_editing_direction
                            if edit_weights:
                                edit_weight_c = edit_weights[c]
                            else:
                                edit_weight_c = 1.0
                            if isinstance(edit_warmup_steps, list):
                                edit_warmup_steps_c = edit_warmup_steps[c]
                            else:
                                edit_warmup_steps_c = edit_warmup_steps

                            if isinstance(edit_cooldown_steps, list):
                                edit_cooldown_steps_c = edit_cooldown_steps[c]
                            elif edit_cooldown_steps is None:
                                edit_cooldown_steps_c = i + 1
                            else:
                                edit_cooldown_steps_c = edit_cooldown_steps
                            if i >= edit_warmup_steps_c:
                                warmup_inds.append(c)
                            if i >= edit_cooldown_steps_c:
                                noise_guidance_edit[c, :, :, :, :] = torch.zeros_like(noise_pred_edit_concept)
                                continue

                            noise_guidance_edit_tmp = noise_pred_edit_concept - noise_pred_uncond
                            # tmp_weights = (noise_pred_text - noise_pred_edit_concept).sum(dim=(1, 2, 3))
                            tmp_weights = (noise_guidance - noise_pred_edit_concept).sum(dim=(1, 2, 3))

                            tmp_weights = torch.full_like(tmp_weights, edit_weight_c)  # * (1 / enabled_editing_prompts)
                            if reverse_editing_direction_c:
                                noise_guidance_edit_tmp = noise_guidance_edit_tmp * -1
                            concept_weights[c, :] = tmp_weights

                            noise_guidance_edit_tmp = noise_guidance_edit_tmp * edit_guidance_scale_c

                            # torch.quantile function expects float32
                            if noise_guidance_edit_tmp.dtype == torch.float32:
                                tmp = torch.quantile(
                                    torch.abs(noise_guidance_edit_tmp).flatten(start_dim=2),
                                    edit_threshold_c,
                                    dim=2,
                                    keepdim=False,
                                )
                            else:
                                tmp = torch.quantile(
                                    torch.abs(noise_guidance_edit_tmp).flatten(start_dim=2).to(torch.float32),
                                    edit_threshold_c,
                                    dim=2,
                                    keepdim=False,
                                ).to(noise_guidance_edit_tmp.dtype)

                            noise_guidance_edit_tmp = torch.where(
                                torch.abs(noise_guidance_edit_tmp) >= tmp[:, :, None, None],
                                noise_guidance_edit_tmp,
                                torch.zeros_like(noise_guidance_edit_tmp),
                            )
                            noise_guidance_edit[c, :, :, :, :] = noise_guidance_edit_tmp

                            # noise_guidance_edit = noise_guidance_edit + noise_guidance_edit_tmp

                        warmup_inds = torch.tensor(warmup_inds).to(device)
                        if len(noise_pred_edit_concepts) > warmup_inds.shape[0] > 0:
                            concept_weights = concept_weights.to("cpu")  # Offload to cpu
                            noise_guidance_edit = noise_guidance_edit.to("cpu")

                            concept_weights_tmp = torch.index_select(concept_weights.to(device), 0, warmup_inds)
                            concept_weights_tmp = torch.where(
                                concept_weights_tmp < 0, torch.zeros_like(concept_weights_tmp), concept_weights_tmp
                            )
                            concept_weights_tmp = concept_weights_tmp / concept_weights_tmp.sum(dim=0)
                            # concept_weights_tmp = torch.nan_to_num(concept_weights_tmp)

                            noise_guidance_edit_tmp = torch.index_select(
                                noise_guidance_edit.to(device), 0, warmup_inds
                            )
                            noise_guidance_edit_tmp = torch.einsum(
                                "cb,cbijk->bijk", concept_weights_tmp, noise_guidance_edit_tmp
                            )
                            noise_guidance_edit_tmp = noise_guidance_edit_tmp
                            noise_guidance = noise_guidance + noise_guidance_edit_tmp

                            del noise_guidance_edit_tmp
                            del concept_weights_tmp
                            concept_weights = concept_weights.to(device)
                            noise_guidance_edit = noise_guidance_edit.to(device)

                        concept_weights = torch.where(
                            concept_weights < 0, torch.zeros_like(concept_weights), concept_weights
                        )

                        concept_weights = torch.nan_to_num(concept_weights)

                        noise_guidance_edit = torch.einsum("cb,cbijk->bijk", concept_weights, noise_guidance_edit)

                        noise_guidance_edit = noise_guidance_edit + edit_momentum_scale * edit_momentum

                        edit_momentum = edit_mom_beta * edit_momentum + (1 - edit_mom_beta) * noise_guidance_edit

                        if warmup_inds.shape[0] == len(noise_pred_edit_concepts):
                            noise_guidance = noise_guidance + noise_guidance_edit

                    if sem_guidance is not None:
                        edit_guidance = sem_guidance[i].to(device)
                        noise_guidance = noise_guidance + edit_guidance

                noise_pred = noise_pred_uncond + noise_guidance

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents
        else:
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
