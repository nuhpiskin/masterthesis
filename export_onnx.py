import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
from models.resnet18_unet import *
from torch.autograd import Variable
import onnx
import os
def load_model(name):

    device = "cuda"
    model = ResNetUNet(1)
    model.load_state_dict(torch.load(f"logs/test2/{name}.pth"))
    model = model.to(device)
    model.eval()
    return model


def torch2onnx():
    simplify = True
    dynamic = True
    name = "model_best"
    model = load_model(name)
    size = 192
    ori_output_file = f"./segmentation_model_{size}x{size}_ONNX.onnx"

    dummy_input = Variable(torch.randn(((1,3,size,size)))).cuda()

    torch.onnx.export(
            model, 
            dummy_input, 
            ori_output_file, 
            opset_version=11,
            input_names = ["img"],
            output_names = ["output_1"],
            dynamic_axes={'img' : {0 : 'batch_size'},    # variable length axes
                            'output_1' : {0 : 'batch_size'}})  


    if simplify :
        model = onnx.load(ori_output_file)
        if simplify:
            from onnxsim import simplify
            if dynamic:
                input_shapes = {model.graph.input[0].name : list((1,3,size,size))}

                model, check = simplify(model, input_shapes=input_shapes, dynamic_input_shape=True)
            else:
                model, check = simplify(model)
            assert check, "Simplified ONNX model could not be validated"
        onnx.save(model, ori_output_file.replace(".onnx","_simpifyed.onnx"))
        os.remove(ori_output_file)
    print("Model is converted.")
if __name__ == "__main__":
    torch2onnx()