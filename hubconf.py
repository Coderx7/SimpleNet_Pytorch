# Optional list of dependencies required by the package
dependencies = ["torch"]

from torchvision.models import get_model_weights, get_weight

from imagenet.simplenet import (
    simplenetv1_5m_m1,
    simplenetv1_5m_m2,
    simplenetv1_9m_m1,
    simplenetv1_9m_m2,
    simplenetv1_small_m1_05,
    simplenetv1_small_m2_05,
    simplenetv1_small_m1_075,
    simplenetv1_small_m2_075,
)
