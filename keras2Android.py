from keras.models import Sequential
from keras.models import model_from_json, load_model
from keras import backend as K
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import os

# Load existing model.
from tensorflow.python.util.compat import as_text

# with open("model/modelArc.json",'r') as f:
#     modelJSON = f.read()

# model=load_model("model/model2.h5")
# model.load_weights("model/modelArc.h5")
model=load_model("Model/model1000-18.h5")
model.load_weights("Model/model_weights-18.h5")

print(model.output)

# All new operations will be in test mode from now on.
K.set_learning_phase(0)

# Serialize the model and get its weights, for quick re-building.
config = model.get_config()
weights = model.get_weights()

# Re-build a model where the learning phase is now hard-coded to 0.
new_model = Sequential.from_config(config)
new_model.set_weights(weights)

temp_dir = "graph"
checkpoint_prefix = os.path.join(temp_dir, "saved_checkpoint")
checkpoint_state_name = "checkpoint_state"
input_graph_name = "input_graph.pb"
output_graph_name = "output_graph.pb"

# Temporary save graph to disk without weights included.
saver = tf.train.Saver()
checkpoint_path = saver.save(K.get_session(), checkpoint_prefix, global_step=0, latest_filename=checkpoint_state_name)
tf.train.write_graph(K.get_session().graph, temp_dir, input_graph_name)

input_graph_path = os.path.join(temp_dir, input_graph_name)
input_saver_def_path = ""
input_binary = False
output_node_names = "dense_4/Softmax" # model dependent
restore_op_name =  "save/restore_all"
filename_tensor_name = "save/Const:0"
output_graph_path = os.path.join(temp_dir, output_graph_name)
clear_devices = False

# Embed weights inside the graph and save to disk.
freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path,
                          output_node_names, restore_op_name,
                          filename_tensor_name, output_graph_path,clear_devices, "")
print("end")


def print_graph_nodes(filename):
    import tensorflow as tf
    g = tf.GraphDef()
    g.ParseFromString(open(filename, 'rb').read())
    print()
    print(filename)
    print("=======================INPUT=========================")
    print([n for n in g.node if n.name.find('input') != -1])
    print("=======================OUTPUT========================")
    print([n for n in g.node if n.name.find('output') != -1])
    print("===================KERAS_LEARNING=====================")
    print([n for n in g.node if n.name.find('keras_learning_phase') != -1])
    print("======================================================")
    print()

print_graph_nodes("graph/Drowsy.pb")