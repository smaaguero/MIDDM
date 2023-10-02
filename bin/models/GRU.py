import tensorflow as tf
import custom_losses


class GRUModel:
    """
    GRUModel class builds and trains a Gated Recurrent Unit (GRU) model
    with specified layers and hyperparameters.
    
    Attributes:
    -----------
    layers : list
        A list containing the size of the input layer and the GRU layer.
    hyperparameters : dict
        A dictionary containing key hyperparameters for model building and training.
        
    Methods:
    --------
    build_model(lr_sch):
        Builds the GRU model with the specified learning rate scheduler.
    train(x_train, y_train, epochs, batch_size, validation_data):
        Trains the built model with the provided training and validation data.
    """
    
    def __init__(self, layers, hyperparameters):
        """
        Initializes the GRUModel with layers and hyperparameters.
        
        Parameters:
        -----------
        layers : list
            A list containing the size of the input layer and the GRU layer.
        hyperparameters : dict
            A dictionary containing key hyperparameters for model building and training.
        """
        self.layers = layers
        self.hyperparameters = hyperparameters
        
    def build_model(self, lr_sch):
        """
        Builds the GRU model with specified learning rate scheduler.
        
        Parameters:
        -----------
        lr_sch : float
            Learning rate for the optimizer during training.
            
        Returns:
        --------
        model : tf.keras.Model
            The compiled GRU model.
        """
        # Define input layer with dynamic shape and masking
        dynamic_input = tf.keras.layers.Input(shape=(self.hyperparameters["timeStep"], self.layers[0]))
        masked = tf.keras.layers.Masking(mask_value=self.hyperparameters['maskValue'])(dynamic_input)
        
        # Define GRU layer with specified parameters
        gru_encoder = tf.keras.layers.GRU(
            self.layers[1],
            dropout=self.hyperparameters['dropout'],
            return_sequences=False,
            activation='tanh',
            use_bias=True
        )(masked)

        # Define output layer with sigmoid activation function
        output = tf.keras.layers.Dense(1, activation="sigmoid")(gru_encoder)
        
        # Compile the model with Adam optimizer and custom loss function
        model = tf.keras.Model(dynamic_input, [output])
        my_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sch)
        customized_loss = custom_losses.weighted_binary_crossentropyv2(self.hyperparameters)
        model.compile(loss=customized_loss, optimizer=my_optimizer)
        
        return model
        
    def train(self, x_train, y_train, epochs, batch_size, validation_data):
        """
        Trains the built model with provided training and validation data.
        
        Parameters:
        -----------
        x_train : numpy array
            Input training data.
        y_train : numpy array
            Target training data.
        epochs : int
            Number of training epochs.
        batch_size : int
            Batch size for training.
        validation_data : tuple
            Tuple containing input and target validation data.
        
        Returns:
        --------
        history : tf.keras.callbacks.History
            A record of training loss values and metrics values at successive epochs.
        model : tf.keras.Model
            The trained GRU model.
        """
        model = self.build_model(lr_sch=self.hyperparameters['learning_rate'])
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
        return history, model