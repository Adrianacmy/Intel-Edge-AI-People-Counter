"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import pprint as pp

from random import randint

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.4,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-op", "--output_path", type=str, default="./", help="Specify the outputpath")
    parser.add_argument("-pc", "--perf_counts", type=bool, help="display performance count", default=True)

    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def draw_boxes(frame, result, width, height, prob):
    '''
    Draw bounding boxes onto the frame.
    '''
    current_count = 0
    duration = 0
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= prob:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 5)
            t1 = time.time()
            if (430 < (xmin+xmax)/2 < 480) and 180 <= (ymin+ymax)/2 <= 210:
                duration += time.time() - t1
            elif (ymin - ymax) < 250 and xmin < 200:
                current_count += 1
    return frame, current_count, duration

def infer_on_stream(args, client):
    print("infer_on_stream...................")
    # Create a black image
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    single_image_mode = False

    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(
        args.model,
        args.device
        )
    ### TODO: Handle the input stream ###
    net_input_shape = infer_network.get_input_shape()['image_tensor']

    if args.input == 'CAM':
        args.input = 0
    elif args.input.endswith('.png') or args.input.endswith('bmp'):
        single_image_mode = True

    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    # Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))
    people_total = 0
    duration = 0
    current_count = 0
    pre_count = 0

    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(args.output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)

    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break

        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]), interpolation = cv2.INTER_AREA)
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        request_id = 0

        t0 = time.time()
        infer_network.exec_net(request_id, {'image_tensor':p_frame} )

        if infer_network.wait() == 0:
#             if args.perf_counts:
#                 pp.pprint(infer_network.requests[0].get_perf_counts())
            t1 = time.time() - t0 #inference time
            result = infer_network.get_output()
            frame, current_count, duration_ = draw_boxes(frame, result, initial_w, initial_h, prob_threshold)

            if current_count > pre_count:
                people_total = people_total + current_count - pre_count
            else:
                duration += duration_

            if people_total > 0:
                client.publish("person/duration", json.dumps({"duration": duration/people_total}))
                client.publish("person", json.dumps({"count": current_count, "total": people_total}))

            pre_count = current_count
            out_video.write(frame)

            print("current count............................", current_count)
            print("total...................................", people_total)

        ### Send the frame to the FFMPEG server ###
        frame = cv2.resize(frame, (768, 432))

        try:
            sys.stdout.buffer.write(frame)
            sys.stdout.flush()
        except BrokenPipeError:
            # Python flushes standard streams on exit; redirect remaining output
            # to devnull to avoid another BrokenPipeError at shutdown
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
            sys.exit(1)

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite("output_image.jpg", frame)
            cv2.imshow('frame', frame)

    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
