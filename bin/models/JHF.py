import tensorflow as tf
import custom_losses


class JHF:
    """
    FHSI class builds and trains a Gated Recurrent Unit (FHSI) model
    with specified hyperparameters.
    
    Attributes:
    -----------
    hyperparameters : dict
        A dictionary containing key hyperparameters for model building and training.
        
    Methods:
    --------
    build_model(hyperparameters):
        Builds the GRU model with the specified learning rate scheduler.
    """

    def __init__(self, hyperparameters):
        """
        Initializes the FHSI with hyperparameters.
        
        Parameters:
        -----------
        hyperparameters : dict
            A dictionary containing key hyperparameters for model building and training.
        """
        self.hyperparameters = hyperparameters


    def build_model(self, hyperparameters):
        # Static preprocessing.
        static_input = tf.keras.layers.Input(shape=(hyperparameters["n_static_features"]))
        hidden_layer = tf.keras.layers.Dense(
            hyperparameters["hidden_layer_size"],
            activation='tanh'
        )(static_input)
        
        # Dynamic preprocessing.
        dynamic_input = tf.keras.layers.Input(shape=(hyperparameters["n_timesteps"], hyperparameters["n_dynamic_features"],))
        masked = tf.keras.layers.Masking(mask_value=666)(dynamic_input)
        gru_encoder = tf.keras.layers.GRU(
            hyperparameters["hidden_layer_size"],
            dropout=hyperparameters["dropout_rate"],
            return_sequences=False,
            activation='tanh',
            use_bias=False
        )(masked)
        
        # Concatenation
        concat_layer = tf.keras.layers.Concatenate()([hidden_layer, gru_encoder])
        output = tf.keras.layers.Dense(1, activation="sigmoid")(concat_layer)
        
        model = tf.keras.Model([static_input, dynamic_input], [output])
        customized_loss = custom_losses.weighted_binary_crossentropy(hyperparameters)
        myOptimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters["lr_scheduler"])
        model.compile(loss=customized_loss, optimizer=myOptimizer)
        
        return model