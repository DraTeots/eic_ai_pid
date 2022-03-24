# This is a short script to test if: 
# 1. Tensorflow is built with CUDA(GPU) support
# 2. GPU is available in the system
import tensorflow as tf

print(f"tf.test.is_built_with_cuda: {tf.test.is_built_with_cuda()}")
print(f"tf.config.list_physical_devices('GPU'): {tf.config.list_physical_devices('GPU')}")
