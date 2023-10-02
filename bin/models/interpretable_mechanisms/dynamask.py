import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf


import random, os, json

class Mask:
    """This class allows to fit and interact with dynamic masks.
    Attributes:
        perturbation (attribution.perturbation.Perturbation):
            An object of the Perturbation class that uses the mask to generate perturbations.
        device: The device used to work with the torch tensors.
        verbose (bool): True is some messages should be displayed during optimization.
        random_seed (int): Random seed for reproducibility.
        deletion_mode (bool): True if the mask should identify the most impactful deletions.
        eps (float): Small number used for numerical stability.
        mask_tensor (torch.tensor): The tensor containing the mask coefficients.
        T (int): Number of time steps.
        N_features (int): Number of features.
        hist (torch.tensor): History tensor containing the metrics at different epochs.
        task (str): "classification" or "regression".
    """

    def __init__(
        self,
        verbose: bool = False,
        random_seed: int = 42,
        deletion_mode: bool = False,
        eps: float = 1.0e-7,
    ):

        self.verbose = verbose
        self.random_seed = random_seed
        self.deletion_mode = deletion_mode
        self.eps = eps
        self.mask_model  = None
        self.T = None
        self.N_features = None
        self.model = None
        self.n_epoch = None
        self.loss_function = None

        
    class FadeMovingAverageWindow(tf.keras.layers.Layer):
        def __init__(self, 
                     w_mask=None,
                     window_size=2,
                     mask_value=666,
                     initial_mask_coeff=1,
                     allow_training=False,
                     **kwargs) :
            self.w_mask = w_mask
            self.window_size = window_size
            self.mask_value = mask_value
            self.initial_mask_coeff = initial_mask_coeff

            self.allow_training = allow_training
            self.supports_masking = True

            super().__init__(**kwargs)
            return

        def get_config(self):
            config = super().get_config().copy()
            config.update({
                'w_mask': self.w_mask,
                'window_size': self.window_size,
                'mask_value': self.mask_value,
                'initial_mask_coeff': self.initial_mask_coeff,
                'allow_training': self.allow_training,
                'supports_masking': self.supports_masking,
            })
            return config

        
        def build(self, input_shape):
            batch_size, nun_time_steps, num_feats = input_shape
            initializer = tf.keras.initializers.glorot_uniform()
            self.w_mask = self.add_weight(shape=(1, nun_time_steps, num_feats),
                                         name='w_mask',
                                         initializer=initializer,
                                         trainable=self.allow_training,
                                         constraint=lambda x: tf.clip_by_value(x, 0, 1))
            super().build(input_shape)
            return

        def call(self, x, mask=None):
            # Masking the missing values with 0s
            if mask != None:
                mask_matrix = tf.transpose((tf.ones([x.shape[2], 1, 1]) * tf.cast(mask, "float")), perm=[1, 2, 0])
                x_masked = tf.where(mask_matrix == 1.0, x, 0.0)
            else:
                x_masked = x.copy()


            #Creating the filter_coefs of the moving average window
            T = x_masked.shape[1]
            T_axis = tf.range(1.0, T + 1, delta=1)
            T1_tensor = tf.expand_dims(T_axis, axis=[1])
            T2_tensor = tf.expand_dims(T_axis, axis=[0])
            filter_coefs = (T1_tensor - T2_tensor)
            filter_coefs = (filter_coefs >= 0) & (filter_coefs <= self.window_size) 
            filter_coefs = tf.cast(filter_coefs, tf.float32)
            filter_coefs = filter_coefs / tf.transpose(tf.reshape(tf.math.reduce_sum(filter_coefs, axis=1), (1, filter_coefs.shape[0])))

            
            
            #Applying the filter of moving average window
            x_avg = tf.linalg.matmul(filter_coefs, x_masked)

            # The perturbation is an affine combination of the input and the previous tensor weighted by the mask
            x_pert = x_avg + self.w_mask * (x_masked - x_avg)


            if mask != None:
                x_pert_masked = tf.where(mask_matrix == 1.0, x_pert, self.mask_value * tf.ones_like(x_pert))
                return x_pert_masked
            else:
                return x_pert 
        
    def compute_loss(
        mask_model,
        T,
        N_features, 
        y_pred_pert, 
        y_pred,
        loss_fn, 
        error_factor,
        reg_factor, 
        time_reg_factor, 
        reg_ref
    ):    
        mask_values = mask_model.get_weights()[0]
        mask_sorted = tf.sort(tf.reshape(mask_values, shape=(T * N_features)))
        size_reg = tf.math.reduce_mean((reg_ref - mask_sorted) ** 2).numpy()
        time_reg = tf.math.reduce_mean(
            tf.abs(mask_values[0, 1:T, :] - mask_values[0, :(T-1), :])
        )
        error = loss_fn(y_pred_pert, y_pred)
        loss = error_factor * error + reg_factor * size_reg + time_reg_factor * time_reg                    
    
        return loss
    
    def build_model(
        initial_mask_coeff, 
        n_time_steps, 
        n_features, 
        mask_value,
        window_size,
        sigma_max=2.0,
        eps=1.0e-7
    ):
        inputs = tf.keras.Input(shape=(n_time_steps, n_features), name="original_input")
        masked = tf.keras.layers.Masking(mask_value=mask_value)(inputs)
        perturbed_output = Mask.FadeMovingAverageWindow(w_mask=None,
                                                        window_size = window_size,
                                                        mask_value = mask_value,
                                                        initial_mask_coeff=initial_mask_coeff,
                                                        allow_training=True)(masked)
        model = tf.keras.Model(inputs=inputs, outputs=[inputs, perturbed_output])
        
        return model



    def check_early_stopping(
        loss_bce, past_loss_bce, min_value, v_loss_bce, v_loss_bce_peek,
        flag_peek, peek_duration, counter, min_delta,
        mask_model
    ):
        stop_training = False
        if flag_peek == False:
            v_loss_bce.append(loss_bce)
            diff = past_loss_bce - loss_bce
            # if the difference betwen the current and past value is higher than delta I stop the train
            if (diff < 0) and (abs(diff) > min_delta):
                flag_peek = True
                # print("Voy a entrar a peek porque:", currentValue, " - ", self.pastValue, " - ", diff)
            else:
                past_loss_bce = loss_bce
                if (loss_bce < min_value):
                    # Guardo el mejor modelo hasta ahora
                    # saveModel(mask_model, "best_model_in_json.json", "best_model_weights.h5")
                    mask_model.save_weights('best_model_in_json.h5')
                    min_value = loss_bce
        else:
            # print("Entra en flag peek...")
            if counter < peek_duration:
                # Guardo los sucesivos modelos
                # title_json = "model_in_json" + str(counter) + ".json"
                # title_h5 = "model_weights" + str(counter) + ".h5"
                # saveModel(mask_model, title_json, title_h5)
                mask_model.save_weights("model_weights" + str(counter) + ".h5")

                # Me guardo los valores de la métrica
                v_loss_bce_peek.append(loss_bce)
                counter = counter + 1
            else:
                flag_peek = False
                # Comparo el valor del mejor modelo con el de los modelos en peek
                best_value = v_loss_bce[-2]
                minimum_peek_value = min(v_loss_bce_peek)
                if minimum_peek_value < best_value:
                    # Añado los valores de la métrica hasta el valor máximo de peek
                    index = v_loss_bce_peek.index(minimum_peek_value)
                    v_loss_bce.extend(v_loss_bce_peek[:(index + 1)])
                    if (minimum_peek_value < min_value):
                        # Guardo el mejor modelo hasta ahora
                        title_h5 = "model_weights" + str(index) + ".h5"
                        mask_model.load_weights(title_h5)
                        # saveModel(mask_model, "best_model_in_json.json", "best_model_weights.h5")
                        mask_model.save_weights('best_model_in_json.h5')
                        
                        min_value = minimum_peek_value
                    # Pongo el contador a 0
                    counter = 0
                    v_loss_bce_peek = []
                else:
                    stop_training = True
        return stop_training, flag_peek,  past_loss_bce, min_value, v_loss_bce, v_loss_bce_peek, counter

    # Mask Optimization
    def fit(
        self,
        x_train,
        x_val,
        model,
        n_epochs: int = 500,
        keep_ratio: float = 0.5,
        initial_mask_coeff: float = 0.5,
        size_reg_factor_init: float = 0.5,
        size_reg_factor_dilation: float = 100,
        time_reg_factor: float = 0,
        
        sigma_max: float = 2.0,
        eps: float = 1.0e-7,
        window_size=2,
        
        mask_value = 666,
        peek_duration = 5,
        min_delta = 0.00001,
        learning_rate: float = 1.0e-3,
        momentum: float = 0.9,
    ):
        """This method fits a mask to the input X for the black-box function f.
        Args:
            X: Input matrix (as a T*N_features torch tensor).
            f: Black-box (as a map compatible with torch tensors).
            target: If the output to approximate is different from f(X), it can be specified optionally.
            n_epoch: Number of steps for the optimization.
            keep_ratio: Fraction of elements in X that should be kept by the mask (called a in the paper).
            initial_mask_coeff: Initial value for the mask coefficient (called lambda_0 in the paper).
            size_reg_factor_init: Initial coefficient for the regulator part of the total loss.
            size_reg_factor_dilation: Ratio between the final and the initial size regulation factor
                (called delta in the paper).
            time_reg_factor: Regulation factor for the variation in time (called lambda_a in the paper).
            learning_rate: Learning rate for the torch SGD optimizer.
            momentum: Momentum for the SGD optimizer.
        Returns:
            None
        """
        self.model = model
        self.n_epochs = n_epochs
        _, self.T, self.N_features = x_train._flat_shapes[0]
        
        reg_factor = size_reg_factor_init
        error_factor = 1 - 2 * self.deletion_mode  # In deletion mode, the error has to be maximized
        reg_multiplicator = np.exp(np.log(size_reg_factor_dilation) / n_epochs)
        # print(reg_multiplicator)
        # Initializing the reference vector used in the size regulator (called r_a in the paper)
        reg_ref = tf.zeros(int((1 - keep_ratio) * self.T * self.N_features))
        reg_ref = tf.concat(
            (reg_ref, tf.ones(self.T * self.N_features - reg_ref.shape[0])), axis=0
        )      
        # The initial mask model is defined with the initial mask coefficient
        mask_model = Mask.build_model(initial_mask_coeff, 
                                      self.T, self.N_features,
                                      window_size=window_size,
                                      mask_value=mask_value)
        # Instantiate an optimizer.
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        # Instantiate a loss function.
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        
        # Run the optimization
        hist_train = []
        hist_bce_train = []
        hist_val = []
        hist_bce_val = []
        hist_bce_val_peek = []

        stop_training = False
        flag_peek = False
        past_loss_bce = np.inf
        min_value = np.inf
        counter=0
        for epoch in range(n_epochs):
            # Iterate over the batches of the dataset.
            v_loss_train = []
            v_loss_bce_train = []
            for step, (x_batch_train, x_batch_train_static, y_batch_train) in enumerate(x_train):
                # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:
                    # Apply the mask_model to correct the perturbations.
                    x_batch, x_batch_pert = mask_model(x_batch_train, training=True)
                    
                    # Get the predictions using the black-box model
                    y_pred_pert = self.model([x_batch_pert, x_batch_train_static], training=False)
                    y_pred = self.model([x_batch_train, x_batch_train_static], training=False)
                                        
                    # Compute the loss value for this batch.
                    loss = Mask.compute_loss(mask_model, 
                                             self.T, self.N_features, 
                                             y_pred_pert, y_pred, 
                                             loss_fn, error_factor, reg_factor, time_reg_factor, reg_ref)
                    v_loss_train.append(loss)
                    
                    loss_bce = loss_fn(y_pred_pert, y_pred)                
                    v_loss_bce_train.append(loss_bce)
                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss, mask_model.trainable_weights)
                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, mask_model.trainable_weights))
            
            v_loss_val = []
            v_loss_bce_val = []
            for step, (x_batch_val, x_batch_val_static, y_batch_val) in enumerate(x_val):
                # Apply the mask_model to correct the perturbations.
                x_batch, x_batch_pert = mask_model(x_batch_val, training=False)
                # Get the predictions using the black-box model
                y_pred_pert = self.model([x_batch_pert, x_batch_val_static], training=False)
                y_pred = self.model([x_batch_val, x_batch_val_static], training=False)
                
                # Compute the loss value for this batch.
                loss = Mask.compute_loss(mask_model, 
                                         self.T, self.N_features, 
                                         y_pred_pert, y_pred, 
                                         loss_fn, error_factor, reg_factor, time_reg_factor, reg_ref)
                v_loss_val.append(loss)
                
                loss_bce = loss_fn(y_pred_pert, y_pred)                
                v_loss_bce_val.append(loss_bce)
                
            self.mask_model = mask_model

            stop_training, flag_peek,  past_loss_bce, min_value, hist_bce_val, hist_bce_val_peek, counter = self.check_early_stopping(
                np.array(v_loss_bce_val).mean(), past_loss_bce, min_value, hist_bce_val, hist_bce_val_peek,
                flag_peek, peek_duration, counter, min_delta,
                mask_model
            )
            
            if stop_training:
                # print("PARO DE ENTRENAR!!!")
                break
            
            # print("reg_factor", reg_factor)
            reg_factor *= reg_multiplicator
            
    
            print(
                "Epoch number ", epoch, 
                ", Training_loss: " ,  np.around(np.array(v_loss_train).mean(), decimals=6),
                ", Val_loss: " ,  np.around(np.array(v_loss_val).mean(), decimals=6),
                sep=""
            )
            hist_train.append(np.array(v_loss_train).mean())
            hist_val.append(np.array(v_loss_val).mean())
            hist_bce_train.append(np.array(v_loss_bce_train).mean())
            # hist_bce_val.append(np.array(v_loss_bce_val).mean())            
        return np.array(hist_train), np.array(hist_val), np.array(hist_bce_train), np.array(hist_bce_val)