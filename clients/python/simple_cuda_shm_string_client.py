#!/usr/bin/env python
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
import os
from builtins import range
from tensorrtserver.api import *
import tensorrtserver.cuda_shared_memory as cudashm
from ctypes import *

FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-i', '--protocol', type=str, required=False, default='http',
                        help='Protocol ("http"/"grpc") used to ' +
                        'communicate with inference service. Default is "http".')
    parser.add_argument('-H', dest='http_headers', metavar="HTTP_HEADER",
                        required=False, action='append',
                        help='HTTP headers to add to inference server requests. ' +
                        'Format is -H"Header:Value".')

    FLAGS = parser.parse_args()
    protocol = ProtocolType.from_str(FLAGS.protocol)

    # We use a simple model that takes 2 input tensors of 16 strings
    # each and returns 2 output tensors of 16 strings each. The input
    # strings must represent integers. One output tensor is the
    # element-wise sum of the inputs and one output is the element-wise
    # difference.
    model_name = "simple_string"
    model_version = -1
    batch_size = 1

    # Create the inference context for the model.
    infer_ctx = InferContext(FLAGS.url, protocol, model_name, model_version, FLAGS.verbose)

    # Create the shared memory control context
    shared_memory_ctx = SharedMemoryControlContext(FLAGS.url, protocol, \
                            http_headers=FLAGS.http_headers, verbose=FLAGS.verbose)

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones. The input tensors
    # are the string representation of these values.
    in0 = np.arange(start=0, stop=16, dtype=np.int32)
    in1 = np.ones(shape=16, dtype=np.int32)
    expected_sum = np.add(in0, in1)
    expected_diff = np.subtract(in0, in1)

    in0n = np.array([str(x) for x in in0.reshape(in0.size)], dtype=object)
    input0_data = in0n.reshape(in0.shape)
    in1n = np.array([str(x) for x in in1.reshape(in1.size)], dtype=object)
    input1_data = in1n.reshape(in1.shape)

    # serialize the string tensors
    input0_data_serialized = serialize_string_tensor(input0_data)
    input1_data_serialized = serialize_string_tensor(input1_data)

    # Use the size of the serialized tensors to create the shared memory regions
    input0_byte_size = input0_data_serialized.size * input0_data_serialized.itemsize
    input1_byte_size = input1_data_serialized.size * input1_data_serialized.itemsize
    output_byte_size = max(input0_byte_size, input1_byte_size) + 1
    output_byte_size = max(input0_byte_size, input1_byte_size) + 1

    # Create Output0 and Output1 in Shared Memory and store shared memory handles
    shm_op0_handle = cudashm.create_shared_memory_region("output0_data", output_byte_size, 0)
    shm_op1_handle = cudashm.create_shared_memory_region("output1_data", output_byte_size, 0)

    # Register Output0 and Output1 shared memory with TRTIS
    shared_memory_ctx.cuda_register(shm_op0_handle)
    shared_memory_ctx.cuda_register(shm_op1_handle)

    # Create Input0 and Input1 in Shared Memory and store shared memory handles
    shm_ip0_handle = cudashm.create_shared_memory_region("input0_data", input0_byte_size, 0)
    shm_ip1_handle = cudashm.create_shared_memory_region("input1_data", input1_byte_size, 0)

    # Put input data values into shared memory
    cudashm.set_shared_memory_region(shm_ip0_handle, [input0_data_serialized])
    cudashm.set_shared_memory_region(shm_ip1_handle, [input1_data_serialized])

    # Register Input0 and Input1 shared memory with TRTIS
    shared_memory_ctx.cuda_register(shm_ip0_handle)
    shared_memory_ctx.cuda_register(shm_ip1_handle)

    # Send inference request to the inference server. Get results for both
    # output tensors. Passing shape of input tensors is necessary for
    # String and variable size tensors.
    results = infer_ctx.run({ 'INPUT0' : (shm_ip0_handle, input0_data.shape),
                            'INPUT1' : (shm_ip1_handle, input1_data.shape)},
                            { 'OUTPUT0' : (InferContext.ResultFormat.RAW, shm_op0_handle),
                            'OUTPUT1' : (InferContext.ResultFormat.RAW, shm_op1_handle) },
                            batch_size)

    # We expect there to be 2 results (each with batch-size 1). Walk
    # over all 16 result elements and print the sum and difference
    # calculated by the model.
    output0_data = results['OUTPUT0'][0]
    output1_data = results['OUTPUT1'][0]

    for i in range(16):
        print(str(input0_data[i]) + " + " + str(input1_data[i]) + " = " + output0_data[i].decode("utf-8"))
        print(str(input0_data[i]) + " - " + str(input1_data[i]) + " = " + output1_data[i].decode("utf-8"))

        # Convert result from string to int to check result
        r0 = int(output0_data[i])
        r1 = int(output1_data[i])
        if expected_sum[i] != r0:
            print("error: incorrect sum");
            sys.exit(1);
        if expected_diff[i] != r1:
            print("error: incorrect difference");
            sys.exit(1);

    print(shared_memory_ctx.get_shared_memory_status())
    shared_memory_ctx.unregister_all()
    cudashm.destroy_shared_memory_region(shm_ip0_handle)
    cudashm.destroy_shared_memory_region(shm_ip1_handle)
    cudashm.destroy_shared_memory_region(shm_op0_handle)
    cudashm.destroy_shared_memory_region(shm_op1_handle)
