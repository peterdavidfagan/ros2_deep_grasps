"""
Converting a gr_convnet model to unified ONNX format.

Important: see README.md for instructions on how to run this script.
"""

import argparse

import torch


# TODO add generalized class that works for different models

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model file")
    parser.add_argument("--output", type=str, help="output file")
    #parser.add_argument("--input_names", type=str, nargs="+", help="input names")
    #parser.add_argument("--output_names", type=str, nargs="+", help="output names")
    args = parser.parse_args()

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model)

    # generate dummy input
    dummy_input = torch.randn(1, 4, 640, 480)
    dummy_input = dummy_input.to(device)

    # Convert to ONNX
    torch.onnx.export(model, 
            dummy_input, 
            args.output,
            input_names=["input"],
            output_names=["output"],
            )

