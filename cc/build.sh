TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared binarized.cc -o binarized.so -fPIC -I $TF_INC


