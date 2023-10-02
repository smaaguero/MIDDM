import tensorflow as tf
import custom_losses

class MLPModel:

    def __init__(self, layers, hyperparameters):
        """
        Initializes the MLPModel with layers and hyperparameters.
        
        Parameters:
        -----------
        layers : list
            A list containing the size of the various layers in the model.
        hyperparameters : dict
            A dictionary containing key hyperparameters for model building and training.
        """
        self.layers = layers
        self.hyperparameters = hyperparameters

    def build_model(self):
        """
        Builds and compiles a multilayer perceptron (MLP) model using the initialized layers and hyperparameters.
        
        Returns:
        --------
        model : tf.keras.Model
            The compiled MLP model.
        """
        # Define Input Layer for static features
        static_input = tf.keras.layers.Input(shape=(self.hyperparameters["n_static_features"]))

        # Define a Hidden Layer with specified neurons and activation function
        hidden_layer = tf.keras.layers.Dense(
            self.hyperparameters["layers_static"],
            activation='tanh'
        )(static_input)
        
        # Apply Dropout for regularization
        dp_layer = tf.keras.layers.Dropout(
            self.hyperparameters["dropout_rate"], 
            noise_shape=None, 
            seed=42
        )(hidden_layer)
        
        # Define Output Layer with sigmoid activation function for binary classification
        output = tf.keras.layers.Dense(1, activation="sigmoid")(dp_layer)
        
        # Compile the Model with Adam optimizer and custom loss function
        model = tf.keras.Model([static_input], [output])
        customized_loss = custom_losses.weighted_binary_crossentropy(self.hyperparameters)
        myOptimizer = tf.keras.optimizers.Adam(learning_rate=self.hyperparameters["learning_rate"])
        model.compile(loss=customized_loss, optimizer=myOptimizer)
        
        return model

# Example Usage:
# layers = [64, 32] # Example layer sizes; adjust as needed
# hyperparameters = { 'n_static_features': 10, 'layers_static': 64, 'dropout_rate': 0.2, 'learning_rate': 0.001 }
# mlp_model = MLPModel(layers, hyperparameters)
# model = mlp_model.build_model()