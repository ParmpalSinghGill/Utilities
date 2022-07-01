# import os
# import os.path as osp
# import argparse
#
# import tensorflow as tf
#
# from keras.models import load_model
# from keras import backend as K
#
#
# def convertGraph(modelPath, outdir, numoutputs, prefix, name):
#     '''
#     Converts an HD5F file to a .pb file for use with Tensorflow.
#     Args:
#         modelPath (str): path to the .h5 file
#            outdir (str): path to the output directory
#        numoutputs (int):
#            prefix (str): the prefix of the output aliasing
#              name (str):
#     Returns:
#         None
#     '''
#
#     # NOTE: If using Python > 3.2, this could be replaced with os.makedirs( name, exist_ok=True )
#     if not os.path.isdir(outdir):
#         os.mkdir(outdir)
#
#     # K.set_learning_phase(0)
#
#     net_model = load_model(modelPath)
#
#     # Alias the outputs in the model - this sometimes makes them easier to access in TF
#     pred = [None] * numoutputs
#     pred_node_names = [None] * numoutputs
#     for i in range(numoutputs):
#         pred_node_names[i] = prefix + '_' + str(i)
#         pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
#     print('Output nodes names are: ', pred_node_names)
#
#     sess = K.get_session()
#
#     # Write the graph in human readable
#     f = 'graph_def_for_reference.pb.ascii'
#     tf.train.write_graph(sess.graph.as_graph_def(), outdir, f, as_text=True)
#     print('Saved the graph definition in ascii format at: ', osp.join(outdir, f))
#
#     # Write the graph in binary .pb file
#     from tensorflow.python.framework import graph_util
#     from tensorflow.python.framework import graph_io
#     constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
#     graph_io.write_graph(constant_graph, outdir, name, as_text=False)
#     print('Saved the constant graph (ready for inference) at: ', osp.join(outdir, name))
#
#
# # if __name__ == '__main__':
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('--model', '-m', dest='model', required=True,
# #                         help='REQUIRED: The HDF5 Keras model you wish to convert to .pb')
# #     parser.add_argument('--numout', '-n', type=int, dest='num_out', required=True,
# #                         help='REQUIRED: The number of outputs in the model.')
# #     parser.add_argument('--outdir', '-o', dest='outdir', required=False, default='./',
# #                         help='The directory to place the output files - default("./")')
# #     parser.add_argument('--prefix', '-p', dest='prefix', required=False, default='k2tfout',
# #                         help='The prefix for the output aliasing - default("k2tfout")')
# #     parser.add_argument('--name', dest='name', required=False, default='output_graph.pb',
# #                         help='The name of the resulting output graph - default("output_graph.pb")')
# #     args = parser.parse_args()
#
# model = load_model("Model/aa/model1000-18.h5")
# model.load_weights("Model/aa/model_weights-18.h5")
# convertGraph("Model/aa/model1000-18.h5", 'Model', 2,'k2tfout', "output_graph.pb")






import os, argparse

import tensorflow as tf

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph

dir = os.path.dirname(os.path.realpath(__file__))


def freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_dir", type=str, default="", help="Model folder to export")
#     parser.add_argument("--output_node_names", type=str, default="",
#                         help="The name of the output nodes, comma separated.")
#     args = parser.parse_args()
#
freeze_graph("keraGraph", "dense_4/Softmax")





# from keras import backend as K
# from keras.models import load_model
# import tensorflow as tf
#
# model = load_model("Model/aa/model1000-18.h5")
# model.load_weights("Model/aa/model_weights-18.h5")
#
# print(model.output.op.name)
# # saver = tf.train.Saver()
# # saver.save(K.get_session(), 'keraGraph/keras_model.ckpt')
#
# # node_names = [node.name for node in tf.get_default_graph.as_graph_def().nod