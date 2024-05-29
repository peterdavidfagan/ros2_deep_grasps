"""Test model inference."""

import numpy as np

import onnx
import onnxruntime as rt

from inference_utils import post_process_output, detect_grasps

def test_inference():
    """Test model inference."""
    model = onnx.load("gr_convnet.onnx")
    onnx.checker.check_model(model)
    sess = rt.InferenceSession("gr_convnet.onnx")
    input_name = "input"
    output_name = None
    input_data = np.random.rand(1, 4, 244, 244).astype(np.float32)
    pred = sess.run(output_name, {input_name: input_data})

    # post process
    q_img, ang_img, width_img = post_process_output(*pred)
    print(q_img.shape, ang_img.shape, width_img.shape)


    # detect grasps
    grasps = detect_grasps(q_img, ang_img, width_img)

    print(grasps)

if __name__ == "__main__":
    test_inference()
