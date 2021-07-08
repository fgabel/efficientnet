from . import inject_tfkeras_modules, init_tfkeras_custom_objects
from . import model

from .preprocessing import center_crop_and_resize

EfficientNetB0 = inject_tfkeras_modules(model.EfficientNetB0)
EfficientNetB1 = inject_tfkeras_modules(model.EfficientNetB1)
EfficientNetB2 = inject_tfkeras_modules(model.EfficientNetB2)
EfficientNetB3 = inject_tfkeras_modules(model.EfficientNetB3)
EfficientNetB4 = inject_tfkeras_modules(model.EfficientNetB4)
EfficientNetB5 = inject_tfkeras_modules(model.EfficientNetB5)
EfficientNetB6 = inject_tfkeras_modules(model.EfficientNetB6)
EfficientNetB7 = inject_tfkeras_modules(model.EfficientNetB7)
EfficientNetL2 = inject_tfkeras_modules(model.EfficientNetL2)
EfficientNetV2B = inject_tfkeras_modules(model.EfficientNetV2B)
EfficientNetV2S = inject_tfkeras_modules(model.EfficientNetV2S)
EfficientNetV2M = inject_tfkeras_modules(model.EfficientNetV2M)
preprocess_input = inject_tfkeras_modules(model.preprocess_input)

init_tfkeras_custom_objects()
