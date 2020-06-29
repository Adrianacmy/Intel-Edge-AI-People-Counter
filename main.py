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
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= 0.5:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
    return frame

def infer_on_stream(args, client):
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

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]), interpolation = cv2.INTER_AREA)
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        request_id = 0
        infer_network.exec_net(request_id, {'image_tensor':p_frame} )

        people_cur = 0 # people in the current frame
        people_total = 0
        duration = 0

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            ### TODO: Get the results of the inference request ###
            ### TODO: Extract any desired stats from the results ###
            result = infer_network.get_output()
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

            for box in result[0][0]:
                conf = box[0]
                if conf >= prob_threshold:
                    xmin = int(box[3] * width)
                    ymin = int(box[4] * height)
                    xmax = int(box[5] * width)
                    ymax = int(box[6] * height)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)

                    if xmin < 400 and ymax < 500:
                        people_cur += 1
                        people_total += 1
                        t1 = time.perf_counter()

                        client.publish('person',json.dumps({
                                        'count': people_cur}))

                    if xmin > 550 and ymax < 400:
                        t2 = time.perf_counter()
                        duration += t2 - t1

                        people_cur = 0

                        client.publish('person',json.dumps({
                                        'count': people_cur}),
                                    qos=0, retain=False)
                        client.publish('person/duration', json.dumps({'duration': duration/people_total}))

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
