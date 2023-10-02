import tensorflow as tf
import sys
sys.path.insert(0, '../')
import custom_losses

class HadamardLayer(tf.keras.layers.Layer):
    def __init__(self,
                allow_training=True,
                l1_penalty=0.0,
                **kwargs):
        
        super(HadamardLayer, self).__init__(**kwargs)
        self.allow_training = allow_training
        self.l1_penalty = l1_penalty
        self.supports_masking = True

    def build(self, input_shape):
        batch_size, num_time_steps, num_feats = input_shape
        self.kernel = self.add_weight(name='kernel', 
                                    shape=(1, num_time_steps, num_feats),
                                    initializer='glorot_uniform',
                                    trainable=self.allow_training, 
                                    regularizer=tf.keras.regularizers.L1(self.l1_penalty))
        super().build(input_shape)

    def call(self, x):
        return x * self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape

class HadamardModel:

    def __init__(self):
        pass

    def build_model(self, layers, hyperparameters, lr_sch, l1_penalty):    
        dynamic_input = tf.keras.layers.Input(shape=(hyperparameters["time_step"], layers[0]))
        masked = tf.keras.layers.Masking(mask_value=hyperparameters['mask_value'])(dynamic_input)

        weighted_tensor = HadamardLayer(allow_training=True, l1_penalty=l1_penalty)(masked)
        
        gru_encoder = tf.keras.layers.GRU(
            layers[1],
            dropout=hyperparameters['dropout_rate'],
            return_sequences=False,
            activation='tanh',
            use_bias=True
        )(weighted_tensor)
        
        output = tf.keras.layers.Dense(1, activation="sigmoid")(gru_encoder)
        model = tf.keras.Model(dynamic_input, [output])
        myOptimizer = tf.keras.optimizers.Adam(learning_rate=lr_sch)
        customized_loss = custom_losses.weighted_binary_crossentropy(hyperparameters)
        model.compile(loss=customized_loss, optimizer=myOptimizer)
            
        return model
