# Project Write-Up

## Explaining Custom Layers
 If a model topology contains any layers are not in the list of known layers, the Model optimizer classifies them as custom

In order to convert any custom layer, you have to add extenstions to both the Model Optimizer and the inference Engine, this will involves Generate custom layer template file, edit the custom layer template files as necessary to create the specialized custom layer extension source code, specify the custom layer extenstion locations to be used by the model optimizer or inference engine.


## Comparing Model Performance

There are many models in TensorFlow 1 Detection Model Zoo, tested
  - ssd_inception_v2_coco_2018_01_28,
  - ssd_mobilenet_v1_coco_2018_01_28
  - faster_rcnn_inception_v2_coco_2018_01_28 <br>
The first two can do the job, but there are too many false positive leads to inaccuracy, the third one has segmentation error.<br>After many tries I choosen to go with
a converted openvino model, person-detection-retail-0013, much higher accuracy an fair latency, after some tunning, confidence threshold 0.58 works good<br><br>
- person-detection-retail-0013
  - download it with `./downloader.py -name person-detection-retail-0013 -o path-to-store`
  - run the app with `python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m person-detection-retail-0013/FP16/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.58 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm`
<pre>
  Time taken to load model..........  1.1139318943023682
  Inferenc time...................... 0.10837841033935547
</pre>

- ssd_inception_v2_coco_2018_01_28
  - download it with
  `wget download it at http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz`,
  - extract it with `tar -xvf sd_inception_v2_coco_2018_01_28.tar.gz`
  - covert this model to IR with below command, this one works well with threshold below 0.6 - 0.7., I am using this one for this project
  <pre>
    python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  </pre>
  - run people counter app with `python main.py -i CAM -m ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.7 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm`
  - before convert to IR
   <pre>
    Time taken to load model....  1.221141529083252
    Inferece time...............  0.2743902220083591
  </pre>
  - after convert to IR
  <pre>
    Time taken to load model....  0.9021141529083252
    Inferece time...............  0.16146588325500488
  </pre>

- ssd_mobilenet_v1_coco_2018_01_28, latency is less, but much less accurate
  - download it with `http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz`
  - extract it with `tar -xvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz`
  - `cd ssd_mobilenet_v1_coco_2018_01_28`
  - convert it to openvino IR with `python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json`
  - run people counter app with `python ../main.py -i CAM -m frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.7 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm`

  - before convert to IR
  <pre>
    Time taken to load model....  0.28904347343484394
    Inferece time...............  0.07431343078613281
  </pre>
  - after convert to IR
  <pre>
    Time taken to load model....  0.22495245933532715
    Inferece time...............  0.05157470703125
  </pre>

- faster_rcnn_inception_v2_coco_2018_01_28,
  - download it with `wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz`
  - extract with `tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz`
  - covert this model to IR with below command, this one seems promising by reading its docs, but it had the segmentation fault(core dumped)
  <pre>
    python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
  </pre>

python main.py -i CAM -m faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.7 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm`


## Assess Model Use Cases

1. Camera monitorning people enter and exit a certain community gate, this is usful to keep the management department informed if there is any unexpected/risky guests inside the neighbourhood

2. Cameras in store checking out waiting line, it could be very helpful to distribute crowds to less busy lines and save time for everyone.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

