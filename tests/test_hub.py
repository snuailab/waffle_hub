from waffle_hub.hub import BaseHub

hub = BaseHub()
print(hub.get_available_backends())
print(hub.is_available_backend("ultralytics", "8.0.25"))
print(hub.is_available_backend("ultralytics", "8.0.5"))
print(hub.is_available_backend("dd", "8.0.5"))
