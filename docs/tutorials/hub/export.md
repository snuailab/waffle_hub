# Export

After you [train `Hub`](train.md), you can now export it using `Hub.export` method.

=== "Python"
    
    ``` python
    from waffle_hub.hub.adapter.ultralytics import UltralyticsHub

    export_callback = hub.export(
        image_size=None,
        batch_size=16,
        opset_version=11,
        hold=True
    )
    ```

    | Argument | Type | Description |
    | --- | --- | --- |
    | image_size | int | export image size. None for same with train_config (recommended). |
    | batch_size | int | max batch size. |
    | opset_version | int | opset version. |
    | hold | bool | hold export. This is an arguement for people using waffle as a SDK. If it is `False`, it will be excecuted by a thread.` |

    | Return | Type | Description |
    | --- | --- | --- |
    | callback | [ExportCallback](../callbacks/#exportcallback) | Callback object |