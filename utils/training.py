import matplotlib
matplotlib.use("Agg")

import numpy as np
import tensorflow as tf
import json as js
import params.param_picker as pp
import models.model_picker as mp
import data.data_selector as ds

def train_mod(data, params, schedule):
    ## Import model
    model = mp.get_model(params['model_type'])
    params["data_shape"] = params["input_shape"]
    model.setup(params, schedule)

    ## Write model weight savers for checkpointing and visualizing graph
    model.write_saver_defs()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=model.graph) as sess:
        ## Need to provide shape if batch_size is used in graph
        sess.run(model.init_op,
            feed_dict={model.x:np.zeros([params["batch_size"]]+params["data_shape"], dtype=np.float32)})

        sess.graph.finalize() # Graph is read-only after this statement
        model.write_graph(sess.graph_def)

        for sch_idx, sch in enumerate(schedule):
            model.sched_idx = sch_idx
            model.log_info("Beginning schedule "+str(sch_idx))
            for b_step in range(model.get_schedule("num_batches")):
                data_batch = data["train"].next_batch(model.batch_size)
                input_data = data_batch[0]
                input_labels = data_batch[1]

                ## Get feed dictionary for placeholders
                feed_dict = model.get_feed_dict(input_data, input_labels)

                # Reset activity from previous batch
                if hasattr(model, "reset_activity"):
                    sess.run([model.reset_activity], feed_dict)

                ## Update weights
                for w_idx in range(len(model.get_schedule("weights"))):
                    sess.run(model.apply_grads[sch_idx][w_idx], feed_dict)

                ## Normalize weights
                if hasattr(model, "norm_weights"):
                    if params["norm_weights"]:
                        sess.run([model.norm_weights], feed_dict)

                ## Generate logs
                current_step = sess.run(model.global_step)
                if (current_step % model.log_int == 0
                    and model.log_int > 0):
                    model.print_update(input_data=input_data, input_labels=input_labels,
                        batch_step=b_step+1)

                ## Plot weights & gradients
                if (current_step % model.gen_plot_int == 0
                    and model.gen_plot_int > 0):
                    model.generate_plots(input_data=input_data, input_labels=input_labels)

                ## Checkpoint
                if (current_step % model.cp_int == 0
                    and model.cp_int > 0):
                    save_dir = model.write_checkpoint(sess)
                    if hasattr(model, "val_on_cp"):
                        if model.val_on_cp: #Compute validation accuracy
                            val_images = data["val"].images
                            val_labels = data["val"].labels
                            with tf.Session(graph=model.graph) as tmp_sess:
                                val_feed_dict = model.get_feed_dict(val_images, val_labels)
                                tmp_sess.run(model.init_op, val_feed_dict)
                                model.weight_saver.restore(tmp_sess,
                                    save_dir+"_weights-"+str(current_step))
                                if hasattr(model, "full_inference"):
                                    sess.run([model.full_inference], val_feed_dict)
                                if hasattr(model, "step_inference"):
                                    for step in range(model.num_steps):
                                        sess.run([model.step_inference], val_feed_dict)
                                val_accuracy = (
                                    np.array(tmp_sess.run(model.accuracy, val_feed_dict)).tolist())
                                stat_dict = {"validation_accuracy":val_accuracy}
                                js_str = js.dumps(stat_dict, sort_keys=True, indent=2)
                                model.log_info("<stats>"+js_str+"</stats>")
        if params['model_type'] == 'lca':
            weights = sess.run([model.phi], feed_dict)
        else: 
            weights = sess.run([model.a], feed_dict)
        np.savez(params["out_dir"]+'weights/'+params["model_name"]+'_weights', weights)
        save_dir = model.write_checkpoint(sess)
        
def infer_coeffs(data, params, schedule):
    ## Import model
    model = mp.get_model(params["model_type"])
    params["data_shape"] = params["input_shape"]
#     schedule[0]["sparse_mult"] = 0.3
    model.setup(params, schedule)
    coeffs = np.zeros((model.get_schedule("num_batches")*model.batch_size, params["num_neurons"]))
    if params["model_type"] == "ica":
        coeffs_squashed = np.zeros((model.get_schedule("num_batches")*model.batch_size, params["num_neurons"]))
    with tf.Session(graph=model.graph) as sess:
        # Need to provide shape if batch_size is used in graph
        sess.run(model.init_op,
            feed_dict={model.x:np.zeros([params["batch_size"]]+params["input_shape"],
            dtype=np.float32)})

        sess.graph.finalize() # Graph is read-only after this statement

        model.sched_idx = 0
        model.log_info("Beginning schedule "+str(model.sched_idx))
        for b_step in np.arange(model.get_schedule("num_batches")):

            ## To prevent random batches for compatibility with tiled images. 
            ## Commented out code below might work for Dylan's updated codebase.
            input_data = data["train"].images[b_step*model.batch_size:(b_step+1)*model.batch_size] 
#             data_batch = data["train"].next_batch(model.batch_size)
#             input_data = data_batch[0]

            ## Get feed dictionary for placeholders
            feed_dict = model.get_feed_dict(input_data)
            model.weight_saver.restore(sess, tf.train.latest_checkpoint(params["cp_dir"]))
            if params["model_type"] == "ica":
                c = sess.run(model.u, feed_dict)
                c_squashed = sess.run(model.z, feed_dict)
                weights = model.a.eval().T
                weights_inv = model.a_inv.eval()
            elif params["model_type"] == "lca":
                c = sess.run(model.a, feed_dict)
                weights = model.phi.eval()    
            else: 
                raise ValueError("Undefined model type")                      
            coeffs[b_step * model.batch_size: (b_step+1) * model.batch_size] = c
            coeffs_squashed[b_step * model.batch_size: (b_step+1) * model.batch_size] = c_squashed

    np.savez(params["out_dir"]+params['model_name']+'_coeffs', coeffs)
    np.savez(params["out_dir"]+params['model_name']+'_weights', weights)
    if params["model_type"] == "ica":
        np.savez(params["out_dir"]+params['model_name']+'_coeffs_squashed', coeffs_squashed)
        np.savez(params["out_dir"]+params['model_name']+'_weights_inv', weights_inv)
    print(params["model_name"]+'_v'+params['version']+' complete')
