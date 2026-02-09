from keras import Model
from keras.src.applications import VGG19


def create_base_model(

) -> Model:

    """Create a VGG19 feature-extraction model for transfer learning.

    This function loads the pretrained VGG19 model (ImageNet weights) and
    constructs a truncated model that outputs the feature map from the
    `block5_conv4` layer. The resulting model can be used as a fixed
    convolutional feature extractor for image captioning or other
    vision tasks.

    Returns:
        A TensorFlow Keras `Model` whose output corresponds to the
        `block5_conv4` layer of VGG19.
    """

    vgg19: Model = VGG19(weights="imagenet")

    print("VGG19 Model Summary ...")
    vgg19.summary()

    print("Create base model from VGG19 as the base model for Transfer Learning later ... ")
    base_model: Model = Model(
        inputs = vgg19.input,
        outputs = vgg19.get_layer("block5_conv4").output
    )

    print("Base Model Summary ...")
    base_model.summary()

    return base_model
