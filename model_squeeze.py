import find_mxnet
import mxnet as mx
import os, sys
from collections import namedtuple

ConvExecutor = namedtuple('ConvExecutor', ['executor', 'data', 'data_grad', 'style', 'content', 'arg_dict'])

def get_symbol():

    model_symbol = mx.symbol.load('squeezenet-symbol.json')
    internals = model_symbol.get_internals()

    # style and content layers
    #style = mx.sym.Group([internals["relu_conv1_output"], internals['fire2_relu_expand1x1_output'], internals['fire2_relu_expand3x3_output'], internals['fire3_relu_squeeze1x1_output'], internals['fire3_relu_expand1x1_output'], internals['fire3_relu_expand3x3_output'], internals['fire4_relu_squeeze1x1_output'], internals['fire4_relu_expand1x1_output'], internals['fire4_relu_expand3x3_output'], internals['fire5_relu_squeeze1x1_output'], internals['fire5_relu_expand1x1_output'], internals['fire5_relu_expand3x3_output'], internals['fire6_relu_squeeze1x1_output'], internals['fire6_relu_expand1x1_output'], internals['fire6_relu_expand3x3_output'], internals['fire7_relu_squeeze1x1_output'], internals['fire7_relu_expand1x1_output'], internals['fire7_relu_expand3x3_output'], internals['fire8_relu_squeeze1x1_output'], internals['fire8_relu_expand1x1_output'], internals['fire8_relu_expand3x3_output'], internals['fire9_relu_squeeze1x1_output'], internals['fire9_relu_expand1x1_output'], internals['fire9_relu_expand3x3_output'], internals['relu_conv1_output']])
    #content = mx.sym.Group([internals["fire2_relu_expand1x1_output"]])
    style = mx.sym.Group([internals["relu_conv1_output"], internals["fire2_relu_expand1x1_output"], internals["fire4_relu_squeeze1x1_output"], internals["fire6_relu_expand3x3_output"]])
    content = mx.sym.Group([internals["relu_conv1_output"]])
    return style, content


def get_executor(style, content, input_size, ctx):
    out = mx.sym.Group([style, content])
    # make executor
    arg_shapes, output_shapes, aux_shapes = out.infer_shape(data=(1, 3, input_size[0], input_size[1]))
    arg_names = out.list_arguments()
    arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=ctx) for shape in arg_shapes]))
    grad_dict = {"data": arg_dict["data"].copyto(ctx)}
    # init with pretrained weight
    pretrained = mx.nd.load("squeezenet-0001.params")
    for name in arg_names:
        if name == "data":
            continue
        key = "arg:" + name
        if key in pretrained:
            pretrained[key].copyto(arg_dict[name])
        else:
            print("Skip argument %s" % name)
    executor = out.bind(ctx=ctx, args=arg_dict, args_grad=grad_dict, grad_req="write")
    return ConvExecutor(executor=executor,
                        data=arg_dict["data"],
                        data_grad=grad_dict["data"],
                        style=executor.outputs[:-1],
                        content=executor.outputs[-1],
                        arg_dict=arg_dict)


def get_model(input_size, ctx):
    style, content = get_symbol()
    return get_executor(style, content, input_size, ctx)
