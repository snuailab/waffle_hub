from typing import Union

import numpy as np
import torch


class ModelWrapper(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        preprocess: callable,
        postprocess: callable,
    ):
        """
        Model Wrapper.
        Use this wrapper when inference, export.

        Args:
            model (torch.nn.Module): model
            preprocess (callable):
                Preprocess Function that
                recieves [batch, channel, height, width] (0~1),
                and
                outputs [batch, channel, height, width].

            postprocess (callable):
                Postprocess Function that
                recieves model raw output,
                and
                outputs results that fit with our convention.

                Classification:
                    [
                        [batch, class_num],
                    ]  # scores per attribute
                Detection:
                    [
                        [batch, bbox_num, 4(x1, y1, x2, y2)],  # bounding box
                        [batch, bbox_num],  # confidence
                        [batch, bbox_num],  # class id
                    ]
                Segmentation:
                    [
                        [batch, bbox_num, 4(x1, y1, x2, y2)],  # bounding box
                        [batch, bbox_num],  # confidence
                        [batch, bbox_num],  # class id
                        [batch, mask(H, W)] # warning: mask size and image size are not same
                    ]
        """
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.postprocess = postprocess

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.preprocess(x)
        x = self.model(x)
        x = self.postprocess(x, image_size=(W, H))
        return x

    def get_layer_names(self) -> list[str]:
        """
        Get all layer names in model.
        """
        return [name for name, _ in self.model.named_modules()]

    def get_layers(self, layer_names: Union[list[str], str]) -> list[torch.nn.Module]:
        """
        Get layer in model by name.

        Args:
            layer_names (Union[list[str], str]): layer names to get

        Returns:
            layers (list[torch.nn.Module]): layers
        """
        if isinstance(layer_names, str):
            layer_names = [layer_names]
        return [layer for name, layer in self.model.named_modules() if name in layer_names]

    def _convert_to_feature_map(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Convert feature to feature map.
        """
        if len(feature.shape) == 4:  # Convolution Feature Map
            return feature
        elif len(feature.shape) == 3:  # Linear(ViT) Feature Map
            batch, dim, channel = feature.shape
            size = int(np.sqrt(dim - 1))
            return feature[:, 1:].view(
                batch, channel, size, size
            )  # TODO: first token is cls token (usually)
        else:
            raise ValueError(f"Unsupported feature map type. {feature.shape}")

    def get_feature_maps(
        self, x, layer_names: Union[list[str], str] = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Get feature maps from model.

        Args:
            x (torch.Tensor): input image
            layer_names (Union[list[str], str]): layer names to get feature maps

        Returns:
            x (torch.Tensor): model output
            feature_maps (dict): feature maps
        """

        feature_maps = {}

        def hook(name):
            def hook_fn(m, i, o):
                feature_maps[name] = o

            return hook_fn

        if layer_names is None:
            layer_names = self.get_layer_names()[-1]
        elif isinstance(layer_names, str):
            layer_names = [layer_names]

        for name, module in self.model.named_modules():
            if name in layer_names:
                module.register_forward_hook(hook(name))

        x = self.forward(x)

        return x, feature_maps

    def get_cam(
        self,
        x: torch.Tensor,
        layer_name: str,
        class_id: int = None,
    ) -> torch.Tensor:
        """
        TODO: experimental
        Get class activation map.

        Args:
            x (torch.Tensor): input image
            layer_name (str): layer name to get feature maps
            class_id (int): class id to get cam
            image_size (tuple[int, int]): image size
            upsample_size (tuple[int, int]): upsample size

        Returns:
            cam (torch.Tensor): class activation map
        """

        activation_list = []
        gradient_list = []

        def hook(name):
            def hook_fn(m, i, o):
                activation_list.append(self._convert_to_feature_map(o).cpu().detach())

                if not hasattr(o, "requires_grad" or not o.requires_grad):
                    return

                def _store_grad(grad):
                    gradient_list.insert(0, self._convert_to_feature_map(grad).cpu().detach())

                o.register_hook(_store_grad)

            return hook_fn

        layer = self.get_layers(layer_name)[0]
        layer.requires_grad_(True)
        handle = layer.register_forward_hook(hook(layer_name))

        self.model.zero_grad()
        output = self.forward(x)[0]
        if class_id is None:
            class_id = torch.argmax(output[0])
        output[:, class_id].backward()

        activations = activation_list[0]
        gradients = gradient_list[0]

        if activations.shape != gradients.shape:
            raise ValueError(
                f"Activation shape {activations.shape} and gradient shape {gradients.shape} are not same."
            )

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)

        # release hooks
        handle.remove()

        # normalize cam to (0, 1)
        cam = torch.nn.functional.relu(cam)
        cam_max = torch.max(cam)
        cam_min = torch.min(cam)
        cam = (cam - cam_min) / (cam_max - cam_min)

        # resize cam
        cam = torch.nn.functional.interpolate(
            cam, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

        return cam
