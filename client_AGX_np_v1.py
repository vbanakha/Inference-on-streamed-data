 

import zmq
import time
import argparse
import numpy as np


import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image

import time
import common_v1

TRT_LOGGER = trt.Logger()


def parse_arguments():
  parser = argparse.ArgumentParser( description='Data Acquisition Process')
  parser.add_argument('--synchronize_subscriber', action='store_true',
      help='Synchronizes this subscriber to publisher (publisher should wait for subscriptions)')
  parser.add_argument('--subscriber_hwm', type=int, default=10*1024, 
      help='Sets high water mark value for this subscriber.')
  parser.add_argument('--publisher_address', default=None,
      help='Remote publisher address')
  parser.add_argument('--publisher_rep_address',
      help='Remote publisher REP address for synchronization') #same as subscriber address
  return parser.parse_args()


def synchronize_subs(context, publisher_rep_address):
  sync_socket = context.socket(zmq.REQ)
  sync_socket.connect(publisher_rep_address)
  sync_socket.send(b'') # Send synchronization signal
  sync_socket.recv() # Receive reply



def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common_v1.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 1, 64, 64]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()





def main():

  args = parse_arguments()

  context = zmq.Context()

  # Subscriber setup
  subscriber_socket = context.socket(zmq.SUB)
  subscriber_socket.set_hwm(args.subscriber_hwm)
  subscriber_socket.connect(args.publisher_address)
  subscriber_socket.setsockopt(zmq.SUBSCRIBE, b'')

  if args.synchronize_subscriber:
    synchronize_subs(context, args.publisher_rep_address)

  onnx_file_path = '/home/anakha/Documents/ONNX/PyTorch/PyTorchPtychoNN.onnx'
  engine_file_path = "//home/anakha/Documents/Pytorch_ONNX.trt"
  input_image_path = "/home/anakha/Documents/X_test_NCHW.npy"
  output_path  = "/home/anakha/Documents/DataStream/Inference_out/"
 

  # Receive images
  total_received=0
  total_size=0
 
  input_resolution = (64,64)
  trt_outputs = []
  output_shapes = [(1, 64, 64, 1), (1, 64, 64, 1)]
  
  with get_engine(onnx_file_path, engine_file_path) as engine:
      inputs, outputs, bindings, stream = common_v1.allocate_buffers(engine) 

  engine = get_engine(onnx_file_path, engine_file_path)
  total_time=0
  i=0
  while True:
    i += 1
    frame = subscriber_socket.recv()
    print('Received frame number {}'.format(i))
    
    if frame == b"end_data": break # End of data acquisition
    total_size += len(frame)
    dims=(64, 64)
    img = np.frombuffer(frame, dtype='float32')
    img = img.reshape(64, 64)
    
    with engine.create_execution_context() as context:
        # Set host input to the image. The common.do_inference_v2 function will copy the input to the GPU before executing.
        time0 = time.time()
        inputs[0].host = img
        
        trt_outputs, time_in_GPU, infer_time, time_out_GPU, time_synch = common_v1.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        
    trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
    np.save(output_path+'{}'.format(i), trt_outputs)
    
    time1 = time.time()
    
    
    print("Time for copying the input to GPU: {}".format(time_in_GPU))
    print("Compute Time: {}".format(infer_time))
    print("Time for copying the output to GPU: {}".format(time_out_GPU))
    print("Time for stream synchronization: {}".format(time_synch))
    
    total_size += len(frame)
    
    total_time +=time1-time0  
    
  print("Rate = {} kB/sec; {} msg/sec".format((total_size/(2**10))/(time1-time0), total_received/(time1-time0)))
  
  print("Time for inference {} s ".format(time1-time0))
  print("Average Time for inference {} s ".format(total_time/3600))

  
if __name__ == '__main__':
    main()
