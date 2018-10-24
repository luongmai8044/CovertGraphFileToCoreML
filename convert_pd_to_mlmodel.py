import tfcoreml as tf_ml_converter
import tensorflow as tf
import os.path
import argparse
import sys
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile


def load_the_tf_graph_definition():
    with open(FLAGS.retrained_graph, 'rb') as f:
        serialized = f.read()
    tf.reset_default_graph()
    original_gdef = tf.GraphDef()
    original_gdef.ParseFromString(serialized)
    return original_gdef

def strip_the_jpeg_decoder_and_pre_processing_part_of_tf_model(input_graph_def):
    input_node_names = ['Mul']
    output_node_names = ['final_result']
    gdef = strip_unused_lib.strip_unused(
            input_graph_def = input_graph_def,
            input_node_names = input_node_names,
            output_node_names = output_node_names,
            placeholder_type_enum = dtypes.float32.as_datatype_enum)
    # Save it to an output file\n",
    with gfile.GFile(FLAGS.strip_retrained_graph, "wb") as f:
        f.write(gdef.SerializeToString())

def convert(strip_retrained_graph):
    frozen_model_file = os.path.abspath(strip_retrained_graph)
    input_tensor_shapes = {"Mul:0":[1,299,299,3]}
    output_tensor_names = ['final_result:0']

    with tf.gfile.GFile(frozen_model_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
    # Then, we import the graph_def into a new Graph
    tf.import_graph_def(graph_def, name="")
    # Convert
    tf_ml_converter.convert(
            tf_model_path=frozen_model_file,
            mlmodel_path=FLAGS.coreml_model_file,
            input_name_shape_dict=input_tensor_shapes,
            output_feature_names=output_tensor_names)

def main(_):
    graph_def = load_the_tf_graph_definition()
    strip_the_jpeg_decoder_and_pre_processing_part_of_tf_model(graph_def)
    convert(FLAGS.strip_retrained_graph)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--retrained_graph',
        type=str,
        default='',
        help='Path to folders of retrained graph.'
    )
    parser.add_argument(
        '--strip_retrained_graph',
        type=str,
        default='',
        help='Path to folders of retrained graph which stripped the JPEG decoder.'
    )
    parser.add_argument(
        '--coreml_model_file',
        type=str,
        default='',
        help='Name CoreML model file.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)