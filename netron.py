
import torch


torch.onnx.export(model, X_test, 'iris.onnx', input_names=["features"], output_names=["logits"])