import matplotlib
matplotlib.use("Agg")

import numpy as np
import tensorflow as tf
import json as js
import models.model_picker as mp
from data.vanHateren import load_vanHateren

## Import parameters & schedules
from params.ica_params import params, schedule

## Get model
model = mp.get_model(params, schedule)
model.write_saver_defs()

## Get data
params["patch_edge_size"] = np.int(np.sqrt(model.num_pixels))
params["rand_state"] = np.random.RandomState(model.rand_seed)
data = load_vanHateren(params)

with tf.Session(graph=model.graph) as sess:
  sess.run(model.init_op,
    feed_dict={model.x:np.zeros((model.num_pixels, model.batch_size),
    dtype=np.float32)}) # Need to provide shape if batch_size is used in graph

  model.write_graph(sess.graph_def)

  for sch_idx, sch in enumerate(schedule):
    model.sched_idx = sch_idx
    model.log_info("Beginning schedule "+str(sch_idx))
    for b_step in range(model.get_sched("num_batches")):
      mnist_batch = data["train"].next_batch(model.batch_size)
      input_images = mnist_batch[0].T
      input_labels = mnist_batch[1].T if mnist_batch[1] is not None else None

      feed_dict = model.get_feed_dict(input_images, input_labels)

      ## Normalize weights
      if params["norm_weights"]:
        sess.run(model.normalize_weights)

      ## Clear activity from previous batch
      if hasattr(model, "clear_activity"):
        sess.run([model.clear_activity], feed_dict)

      ## Run inference
      if hasattr(model, "full_inference"): # all steps in a single op
        sess.run([model.full_inference], feed_dict)
      if hasattr(model, "step_inference"): # op only does one step
        for step in range(model.num_steps):
          sess.run([model.step_inference], feed_dict)

      ## Update weights
      for w_idx in range(len(model.get_sched("weights"))):
        sess.run(model.apply_grads[sch_idx][w_idx], feed_dict)

      ## Generate logs
      current_step = sess.run(model.global_step)
      if (current_step % model.log_int == 0
        and model.log_int > 0):
        model.print_update(input_data=input_images, input_labels=input_labels,
          batch_step=b_step+1)

      ## Plot weights & gradients
      if (current_step % model.gen_plot_int == 0
        and model.gen_plot_int > 0):
        model.generate_plots(input_data=input_images, input_labels=input_labels)

      ## Checkpoint
      if (current_step % model.cp_int == 0
        and model.cp_int > 0):
        save_dir = model.write_checkpoint(sess)

  save_dir = model.write_checkpoint(sess)
  print("Training Complete\n")
