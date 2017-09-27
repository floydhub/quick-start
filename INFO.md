# Restore Models

## Old Style

### Training

File: `mnist_cnn_train.py`.
At the end of the training, before ending the session:

```bash
	# Save model, with weight and graph
	print("Saving the model at /output/mnist_model")
    saver = tf.train.Saver(save_relative_paths=True)
    saver.save(sess, '/output/mnist_model')
```

### Evaluating

File: `mnist_cnn_eval.py`.
Steps to load a pretrained model:

1. Build Graph
2. Define Saver
3. Init Variables
4. Load from last ckp

```bash
	# Restore model weights from previously saved model
    saver.restore(sess, "/output/mnist_model")
    print("Model restored from file: %s" % "/output/mnist_model")
```

## Old Style v2

### Training

File: `mnist_cnn_train.py`.
Same as above.

### Evaluating

File: `mnist_cnn_eval2.py`.
Steps to load a pretrained model:

1. Load the Graph from the `.meta` file
2. Load the weights from the last checkpoint
3. Retrieve tensor placeholder by name
4. Run your computation

```bash
    # Import Graph(with weights) and load run new ops from placeholders
    saver = tf.train.import_meta_graph('/output/mnist_model.meta')
    # Import weights
    saver.restore(sess,tf.train.latest_checkpoint('/output/'))
    # Get placeholder tensor by name
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
```

## New Style use SavedModel API

### Training

File: `mnist_cnn_train2.py`.
Before opening a Tf.Sess init, then inside the Session add graph, variables and after the session save everything.

```bash
builder = tf.saved_model.builder.SavedModelBuilder("/output/")
with tf.Session() as sess:
	...
	builder.add_meta_graph_and_variables(sess, ["EVALUATING"])
# After sess
builder.save()
```

### Evaluating

File: `mnist_cnn_eval3.py`.
Steps to load a pretrained model:

1. Load the Graph and weights with SaveModel specifing model_tag_name and folder
2. Retrieve tensor placeholder by name
3. Run your computation

```bash
	tf.saved_model.loader.load(sess, ["EVALUATING"], "/output/")
    # Retrieve placeholder tensor by name
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
```
