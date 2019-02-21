# orion_recognation
Benoit wraps the tensorflow detection api(https://github.com/tensorflow/models/tree/master/research/object_detection) with ros.

## Dependencies

In this part, we install all the required dependencies step by step.

``` bash
pip install --upgrade pip
pip install msgpack
# For CPU
pip install tensorflow
# For GPU
pip install tensorflow-gpu
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
pip install --user Cython
pip install --user contextlib2
pip install --user jupyter
pip install --user matplotlib
pip install --user pillow
pip install --user lxml
pip install pyyaml
pip install rospkg
# Protobuf Compilation
protoc object_detection/protos/*.proto --python_out=.
```

