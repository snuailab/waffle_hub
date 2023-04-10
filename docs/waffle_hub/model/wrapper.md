# Model Wrapper

Model Wrapper provides a unified interface for same task models independent of the framework.

## Example

=== "Python"
    
    ``` python
    from waffle_hub.hub.adapter.ultralytics import UltralyticsHub
    
    hub = UltralyticsHub.load("digit_detector")
    model = hub.get_model()

    import torch 
    x = torch.rand(1, 3, 224, 224)
    
    # forward
    prediction = model(x)

    # get feature maps
    layer_names = model.get_layer_names()
    prediction, feature_maps = model.get_feature_maps(x, layer_names)
    ```

## Output Format

### Prediction

=== "Classification"

    ``` python
    [
        [batch, class_num],
    ]  # scores per attribute
    ```

=== "Object Detection"

    ``` python
    [
        [batch, bbox_num, 4(x1, y1, x2, y2)],  # bounding box
        [batch, bbox_num],  # confidence
        [batch, bbox_num],  # class id
    ]
    ```

### Feature Maps

=== "Python"

    ``` python
    {
        "layer_name": [batch, ...],
    }
    ```