from qkeras.autoqkeras import *
#from qkeras.autoqkeras import AutoQKHyperModel, AutoQKerasScheduler, AutoQKeras
#from functions import custom_loss_negative, custom_loss_training, roc_objective
from functions_dist import custom_loss_negative, custom_loss_training, roc_objective #ROC_OBJECTIVE REQUIRES NUMPY HENCE EAGER RUN

from qkeras.autoqkeras.forgiving_metrics import forgiving_factor  # pylint: disable=line-too-long
import kerastuner as kt
from kerastuner.tuners import BayesianOptimization, Hyperband, RandomSearch
#tf.compat.v1.enable_eager_execution()

class Custom_AutoQKHyperModel(AutoQKHyperModel):
    def __init__(self, model, metrics, X_test, bsm_data, custom_objects=None, target=None,\
          transfer_weights=False, frozen_layers=None, activation_bits=4, limit=None,\
          tune_filters="none", tune_filters_exceptions=None,\
          layer_indexes=None, learning_rate_optimizer=False,\
          head_name=None, quantization_config=None, extend_model_metrics=True):
        super().__init__(model, metrics, custom_objects, target,\
          transfer_weights, frozen_layers, activation_bits, limit,\
          tune_filters, tune_filters_exceptions,\
          layer_indexes, learning_rate_optimizer,\
          head_name, quantization_config, extend_model_metrics)
        
        # added
        self.X_test = X_test
        self.bsm_data = bsm_data
    
    def build(self, hp):
        """Builds hyperparameterized quantized model."""

        self.groups = {}

        q_model, _ = self.quantize_model(hp)

        if self.learning_rate_optimizer:
            print("... freezing layers {}.".format(", ".join(self.frozen_layers)))
        for layer_name in self.frozen_layers:
            o_weights = self.model.get_layer(layer_name).get_weights()
            layer = q_model.get_layer(layer_name)
            # layer.trainable = False
            weights = layer.get_weights()
            equal_layer = True
            for w in range(len(o_weights)):
                if o_weights[w].shape != weights[w].shape:
                    equal_layer = False
                    break
            if equal_layer:
                layer.set_weights(o_weights)

        self.trial_size = self.target.get_trial(q_model)

        delta = self.target.delta()

        if not self.metrics:
            score_metric = None
        elif isinstance(self.metrics, dict):
            if not self.head_name:
                score_key = list(self.metrics.keys())[0]
            else:
                score_key = self.head_name
            score_metric = self.metrics[score_key]
            if isinstance(score_metric, list):
                score_metric = score_metric[0]
        elif isinstance(self.metrics, list):
            score_metric = roc_objective(q_model, self.X_test, self.bsm_data) # changed
            #self.metrics.append(score_metric)   # append roc_objective to metrics

        self.score = Custom_AutoQKHyperModel.custom_adjusted_score(
            self, delta, score_metric)

        total_factor = self.target.get_total_factor()
        delta_lr = 1.0 + (total_factor < 0) * total_factor


        lr = float(self.model.optimizer.lr.numpy())

        if self.learning_rate_optimizer:
            lr_range = list(lr * np.linspace(delta_lr, 1.1, 5))
            lr_choice = hp.Choice("learning_rate", lr_range)
            self.model.optimizer.learning_rate = lr_choice
        else:
            lr_choice = lr
            print("learning_rate: {}".format(lr))

        optimizer = self.model.optimizer

        q_model.summary()

        metrics = self.metrics

        if self.extend_model_metrics:
            ext_metrics = copy.deepcopy(metrics)
            if isinstance(ext_metrics, dict):
                if not self.head_name:
                    score_key = list(ext_metrics.keys())[0]
                else:
                    score_key = self.head_name
                score_metric = ext_metrics[score_key]
                if isinstance(score_metric, list):
                    score_metric += [self.trial_size_metric(self.trial_size), self.score]
                else:
                    score_metric = [score_metric]
                    score_metric += [self.trial_size_metric(self.trial_size), self.score]
                ext_metrics[score_key] = score_metric
            else:
                ext_metrics += [
                    self.trial_size_metric(self.trial_size),
                    self.score]
            metrics = ext_metrics

        q_model.compile(
            optimizer=optimizer,
            loss=self.model.loss,
            metrics=metrics,
            run_eagerly=True # added ##THIS IS THE PROBLEM it slows down everything else##
        )
        self.q_model = q_model

        self.target.print_stats()
        print_qmodel_summary(q_model)

        return q_model
    
    @staticmethod
    def custom_adjusted_score(hyper_model, delta, metric_function=None):
        def custom_score(y_true, y_pred):
            y_t_rank = len(y_true.shape.as_list())
            y_p_rank = len(y_pred.shape.as_list())
            y_t_last_dim = y_true.shape.as_list()[-1]
            y_p_last_dim = y_pred.shape.as_list()[-1]

            is_binary = y_p_last_dim == 1
            is_sparse_categorical = (
              y_t_rank < y_p_rank or y_t_last_dim == 1 and y_p_last_dim > 1)

            metric = metric_function(y_true, y_pred)
            print(metric)
            return K.cast(metric * (1.0 + delta), K.floatx())
        if not metric_function:
            metric_function = "accuracy"
        return custom_score
    
class Custom_AutoQKeras_class(AutoQKeras):
    def __init__(
          self, model, X_test, bsm_data, metrics=None, custom_objects=None, goal=None,\
          output_dir="result", mode="random", custom_tuner=None,\
          transfer_weights=False, frozen_layers=None, activation_bits=4,\
          limit=None, tune_filters="none",\
          tune_filters_exceptions=None, learning_rate_optimizer=False,\
          layer_indexes=None, quantization_config=None, overwrite=True,\
          head_name=None, score_metric=None, direction_objective="max", **tuner_kwargs):
        
        super().__init__(model, metrics, custom_objects, goal,\
              output_dir, mode, custom_tuner,\
              transfer_weights, frozen_layers, activation_bits,\
              limit, tune_filters,\
              tune_filters_exceptions, learning_rate_optimizer,\
              layer_indexes, quantization_config, overwrite,\
              head_name, score_metric, **tuner_kwargs)
        
        # added
        self.X_test = X_test
        self.bsm_data = bsm_data
        self.direction_objective = direction_objective
        
        autoqkeras_input_args = locals()
        
        if not metrics:
            metrics = []

        if not custom_objects:
            custom_objects = {}

        # goal: { "type": ["bits", "energy"], "params": {...} } or ForgivingFactor
        #   type
        # For type == "bits":
        #   delta_p: increment (in %) of the accuracy if trial is smaller.
        #   delta_n: decrement (in %) of the accuracy if trial is bigger.
        #   rate: rate of decrease/increase in model size in terms of bits.
        #   input_bits; size of input tensors.
        #   output_bits; size of output tensors.
        #   stress: parameter to reduce reference size to force tuner to
        #     choose smaller models.
        #   config: configuration on what to compute for each layer
        #     minimum configuration is { "default": ["parameters", "activations"] }

        if not goal:
            goal = {
              "type": "bits",
              "params": {
                  "delta_p": 8.0,
                  "delta_n": 8.0,
                  "rate": 2.0,
                  "stress": 1.0,
                  "input_bits": 8,
                  "output_bits": 8,
                  "ref_bits": 8,
                  "config": {
                      "default": ["parameters", "activations"]
                  }
              }
            }

        self.overwrite = overwrite
        self.head_name = head_name

        if not isinstance(goal, ForgivingFactor):
            target = forgiving_factor[goal["type"]](**goal["params"])
        else:
            target = goal
            
        if not metrics:
            metrics = ["acc"]
    
        # changed to custom class
        self.hypermodel = Custom_AutoQKHyperModel(model, metrics, self.X_test, self.bsm_data, custom_objects, target,
                    transfer_weights=transfer_weights,
                    frozen_layers=frozen_layers,
                    activation_bits=activation_bits,
                    limit=limit,
                    tune_filters=tune_filters,
                    tune_filters_exceptions=tune_filters_exceptions,
                    layer_indexes=layer_indexes,
                    learning_rate_optimizer=learning_rate_optimizer,
                    head_name=head_name,
                    quantization_config=quantization_config
        )
        
        
        idx = 0
        name = output_dir
        if self.overwrite:
            while os.path.exists(name):
                idx += 1
                name = output_dir + "_" + str(idx)
        output_dir = name
        self.output_dir = output_dir
        
        if score_metric is None:
            if self.head_name:
                score_metric = "val_" + self.head_name + "_score"
            else:
                score_metric = "val_score"
        assert mode in ["random", "bayesian", "hyperband"]
        
        if custom_tuner is not None:
            self.tuner = custom_tuner(
                self.hypermodel,
                autoqkeras_config=autoqkeras_input_args,
                objective=kt.Objective(score_metric, self.direction_objective), # changed
                project_name=output_dir,
                overwrite=True,
                #distribution_strategy=tf.distribute.MirroredStrategy(), #changed
                **tuner_kwargs)
        elif mode == "random":
            self.tuner = RandomSearch(
                self.hypermodel,
                objective=kt.Objective(score_metric, self.direction_objective),# changed
                project_name=output_dir,
                overwrite=True,
                #distribution_strategy=tf.distribute.MirroredStrategy(), # changed
                **tuner_kwargs)
        elif mode == "bayesian":
            self.tuner = BayesianOptimization(
                self.hypermodel,
                objective=kt.Objective(score_metric, self.direction_objective),# changed
                project_name=output_dir,
                overwrite=True,
                #distribution_strategy=tf.distribute.MirroredStrategy(), #changed
                **tuner_kwargs)
        elif mode == "hyperband":
            self.tuner = Hyperband(
                self.hypermodel,
                objective=kt.Objective(score_metric, self.direction_objective),# changed
                project_name=output_dir,
                overwrite=True,
                #distribution_strategy=tf.distribute.MirroredStrategy(), # changed
                **tuner_kwargs)
        else:
            pass

        self.tuner.search_space_summary()

class Custom_AutoQKerasScheduler(AutoQKerasScheduler):
    def __init__(self, model, X_test, bsm_data, metrics=None, custom_objects=None, goal=None,
          output_dir="result", mode="random", transfer_weights=False,
          activation_bits=4, limit=None, tune_filters="none",
          tune_filters_exceptions=None, layer_indexes=None,
          learning_rate_optimizer=False, blocks=None, schedule_block="sequential",
          quantization_config=None, overwrite=True, debug=False, head_name=None, direction_objective="max",
          **tuner_kwargs):
        
        super().__init__(model, metrics, custom_objects, goal,
          output_dir, mode, transfer_weights,
          activation_bits, limit, tune_filters,
          tune_filters_exceptions, layer_indexes,
          learning_rate_optimizer, blocks, schedule_block,
          quantization_config, overwrite, debug, head_name,
          **tuner_kwargs)
        # added
        self.X_test = X_test
        self.bsm_data = bsm_data
        self.direction_objective = direction_objective
        
    def fit(self, *fit_args, **fit_kwargs):
        """Invokes tuner fit algorithm."""

        self.history = []
        self.compute_block_costs(self.blocks, self.model)

        if self.tuner_kwargs.get("max_trials", None):
            max_trials = float(self.tuner_kwargs["max_trials"])

        lr = self.model.optimizer.lr.numpy()

        model = self.model

        frozen_layers = []

        for i, (pattern, cost) in enumerate(self.retrieve_max_block()):

            # now create new limit pattern
            if not self.overwrite:
                if i < self.next_block:
                    print("Resume tuning. Skipping block ", i)
                    continue

            print("... block cost: {:.0f} / {:.0f}".format(cost, self.reference_size))

            if self.tuner_kwargs.get("max_trials", None):
                self.tuner_kwargs["max_trials"] = int(
                    max(10, max_trials * cost / self.reference_size))
                print("... adjusting max_trials for this block to {}".format(
                    self.tuner_kwargs["max_trials"]))

            limit = self.get_limit(model, pattern)
            new_frozen_layers = self.grouped_patterns[pattern]

            assert limit

            print("Pattern {} is : {}".format(i, limit))

            if self.debug:
                frozen_layers = frozen_layers + new_frozen_layers
                continue

            # changed to custom class
            self.autoqk = Custom_AutoQKeras_class(
              model, self.X_test, self.bsm_data, self.metrics,
              custom_objects=self.custom_objects,
              goal=self.target,
              output_dir=self.output_dir + "/" + str(i),
              mode=self.mode,
              transfer_weights=self.transfer_weights,
              frozen_layers=frozen_layers,
              activation_bits=self.activation_bits,
              limit=limit,
              tune_filters=self.tune_filters,
              tune_filters_exceptions=self.tune_filters_exceptions,
              layer_indexes=self.layer_indexes,
              learning_rate_optimizer=self.learning_rate_optimizer,
              quantization_config=self.quantization_config,
              overwrite=self.overwrite,
              head_name=self.head_name,
               direction_objective=self.direction_objective,
              **self.tuner_kwargs)

            self.autoqk.fit(*fit_args, **fit_kwargs)

            self.autoqk.tuner.results_summary()

            self.history.append(self.autoqk.history())

            model = self.autoqk.get_best_model()
            self.learning_rate = model.optimizer.lr.numpy()

            model.compile(
              model.optimizer,
              loss=self.model.loss,
              metrics=self.model.metrics)

            frozen_layers = frozen_layers + new_frozen_layers

            filename = self.output_dir + "/model_block_" + str(i)
            model.save(filename)
            self.next_block = i + 1

            with tf.io.gfile.GFile(os.path.join(self.output_dir, "scheduler.json"),
                                 "w") as f:
                f.write(json.dumps({"next_block": self.next_block}))

        if self.debug:
            return

        self.best_model = model

        for layer_name in frozen_layers:
            layer = model.get_layer(layer_name)
            layer.trainable = True 
        