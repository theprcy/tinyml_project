"""
Name: pi_model_utils.py

Description: This is a helper functions for loading and running inference on the four model
variants (FP32, PTQ, QAT, Binary) using TensorFlow Lite Runtime on the RPi 4B

Handles the difference between int8 models (PTQ, QAT which need input
quantisation before inference) and float32 models (baseline, Binary which do not need it)
"""

import numpy as np

try:
    # ai-edge-litert - Google's replacement option for tflite-runtime
    from ai_edge_litert.interpreter import Interpreter as TFLiteInterpreter
    def make_interpreter(model_path):
        return TFLiteInterpreter(model_path=model_path)
    print("Using ai_edge_litert")
except ImportError:
    try:
        # tflite-runtime in case that the ai-edge-litert not working
        import tflite_runtime.interpreter as tflite
        def make_interpreter(model_path):
            return tflite.Interpreter(model_path=model_path)
        print("Using tflite_runtime")
    except ImportError:
        # to import full TensorFlow in case the above not working
        import tensorflow as tf
        def make_interpreter(model_path):
            return tf.lite.Interpreter(model_path=model_path)
        print("Using tensorflow fallback")


class PiModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        self.is_int8 = self.input_details['dtype'] == np.int8
        if self.is_int8:
            self.input_scale, self.input_zero_point = self.input_details['quantization']

    def _prepare_input(self, image):
        """image: numpy array, shape (32, 32, 3), float32, range [0, 1]"""
        x = np.expand_dims(image.astype(np.float32), axis=0)

        if self.is_int8:
            x = x / self.input_scale + self.input_zero_point
            x = np.clip(np.round(x), -128, 127).astype(np.int8)

        return x

    def predict(self, image):
        """Runs a single inference. Returns the predicted class index"""
        x = self._prepare_input(image)
        self.interpreter.set_tensor(self.input_details['index'], x)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details['index'])
        return int(np.argmax(output[0]))

    def predict_batch(self, images):
        """
	- Runs inference one-by-one over a batch (TFLite interpreters here
        are configured for single-image input)
	
	- Returns a list of predicted
        class indices"""
        return [self.predict(img) for img in images]


MODEL_PATHS = {
    "FP32": "models/baseline.tflite",
    "PTQ": "models/ptq_model.tflite",
    "QAT": "models/qat_model.tflite",
    "Binary": "models/binary_model.tflite",
}


def load_all_models(model_dir="models"):
    """To load all models"""
    models = {}
    for name, path in MODEL_PATHS.items():
        full_path = f"{model_dir}/{path.split('/')[-1]}"
        try:
            models[name] = PiModel(full_path)
            print(f"Loaded {name} from {full_path}")
        except Exception as e:
            print(f"Could not load {name} from {full_path}: {e}")
    return models
