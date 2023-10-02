import tensorflow as tf
import custom_losses

concat = tf.keras.backend.concatenate
stack = tf.keras.backend.stack
K = tf.keras.backend
Dense = tf.keras.layers.Dense
Add = tf.keras.layers.Add
LayerNorm = tf.keras.layers.LayerNormalization
Multiply = tf.keras.layers.Multiply
Dropout = tf.keras.layers.Dropout
Activation = tf.keras.layers.Activation
Lambda = tf.keras.layers.Lambda


class FHSI:
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


    # Layer utility functions.
    def linear_layer(self,
                     size,
                    activation=None,
                    use_time_distributed=False,
                    use_bias=True):
        """Returns simple Keras linear layer.
        Args:
            size: Output size
            activation: Activation function to apply if required
            use_time_distributed: Whether to apply layer across time
            use_bias: Whether bias should be included in layer
        """
        linear = tf.keras.layers.Dense(size, activation=activation, use_bias=use_bias)
        if use_time_distributed:
            linear = tf.keras.layers.TimeDistributed(linear)
        return linear

    def add_and_norm(self, x_list):
        """Applies skip connection followed by layer normalisation.

    Args:
        x_list: List of inputs to sum for skip connection

    Returns:
        Tensor output from layer.
    """
        tmp = Add()(x_list)
        tmp = LayerNorm()(tmp)
        return tmp

    def apply_gating_layer(self, 
                           x,
                        hidden_layer_size,
                        dropout_rate=None,
                        use_time_distributed=True,
                        activation=None):
        """Applies a Gated Linear Unit (GLU) to an input.

    Args:
        x: Input to gating layer
        hidden_layer_size: Dimension of GLU
        dropout_rate: Dropout rate to apply if any
        use_time_distributed: Whether to apply across time
        activation: Activation function to apply to the linear feature transform if
        necessary

    Returns:
        Tuple of tensors for: (GLU output, gate)
    """

        if dropout_rate is not None:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

        if use_time_distributed:
            activation_layer = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(hidden_layer_size, activation=activation))(
                x)
            gated_layer = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'))(
                x)
        else:
            activation_layer = tf.keras.layers.Dense(
                hidden_layer_size, activation=activation)(
                x)
            gated_layer = tf.keras.layers.Dense(
                hidden_layer_size, activation='sigmoid')(
                x)

        return tf.keras.layers.Multiply()([activation_layer,
                                        gated_layer]), gated_layer



    def gated_residual_network(
            self, 
            x,
            hidden_layer_size,
            output_size=None,
            dropout_rate=None,
            use_time_distributed=True,
            additional_context=None,
            return_gate=False
            ):
        """Applies the gated residual network (GRN) as defined in paper.

        Args:
            x: Network inputs
            hidden_layer_size: Internal state size
            output_size: Size of output layer
            dropout_rate: Dropout rate if dropout is applied
            use_time_distributed: Whether to apply network across time dimension
            additional_context: Additional context vector to use if relevant
            return_gate: Whether to return GLU gate for diagnostic purposes

        Returns:
            Tuple of tensors for: (GRN output, GLU gate)
        """
        # Setup skip connection
        if output_size is None:
            output_size = hidden_layer_size
            skip = x
        else:
            linear = Dense(output_size)
            if use_time_distributed:
                linear = tf.keras.layers.TimeDistributed(linear)
            skip = linear(x)

        # Apply feedforward network
        hidden = self.linear_layer(
            hidden_layer_size,
            activation=None,
            use_time_distributed=use_time_distributed)(
            x)
        if additional_context is not None:
            hidden = hidden + self.linear_layer(
                hidden_layer_size,
                activation=None,
                use_time_distributed=use_time_distributed,
                use_bias=False)(
                additional_context)
        hidden = tf.keras.layers.Activation('elu')(hidden)
        hidden = self.linear_layer(
            hidden_layer_size,
            activation=None,
            use_time_distributed=use_time_distributed)(
            hidden)

        gating_layer, gate = self.apply_gating_layer(
            hidden,
            output_size,
            dropout_rate=dropout_rate,
            use_time_distributed=use_time_distributed,
            activation=None)

        if return_gate:
            return self.add_and_norm([skip, gating_layer]), gate
        else:
            return self.add_and_norm([skip, gating_layer])





    def get_static_embeddings(
            self,
            static_input,
                            static_size,
                            category_counts,
                            hidden_layer_size):
        # define the sizes and split the categorical and numerical features
        num_categorical_variables = len(category_counts)
        num_regular_variables = static_size - num_categorical_variables
        regular_inputs, categorical_inputs = static_input[:, :num_regular_variables], \
                                            static_input[:, num_regular_variables:]
        embedding_sizes = [hidden_layer_size for i, size in enumerate(category_counts)]

        # Apply the embeddings to the categorical features
        embedded_inputs = []
        for i in range(num_categorical_variables):
            embedding = tf.keras.layers.Embedding(
                category_counts[i],
                embedding_sizes[i],
                dtype=tf.float32
            )(categorical_inputs[Ellipsis, i])
            embedded_inputs.append(embedding)

        # Stack the categorical features after the embedding and the transformed numerical features
        static_inputs_tr = tf.keras.backend.stack(
            [tf.keras.layers.Dense(hidden_layer_size)(regular_inputs) for i in range(num_regular_variables)] + \
            [embedded_inputs[i] for i in range(num_categorical_variables)],
            axis=1
        )

        return static_inputs_tr


    def static_combine_and_mask(
            self,
            embedding,
                                hidden_layer_size,
                                dropout_rate=None
                                ):
        # Flatten the static features
        _, num_static, _ = embedding.get_shape().as_list()
        flatten = tf.keras.layers.Flatten(name="flattened_inputs")(embedding)

        # Create the weights of the static features passing the input through a GRN, and then a softmax.
        mlp_outputs = self.gated_residual_network(
            flatten,
            hidden_layer_size,
            output_size=num_static,
            dropout_rate=dropout_rate,
            use_time_distributed=False,
            additional_context=None
        )
        sparse_weights = tf.keras.layers.Activation('softmax', name="softmax_act")(mlp_outputs)
        sparse_weights = K.expand_dims(sparse_weights, axis=-1)

        # Transform the input through a GRN.
        trans_emb_list = []
        for i in range(num_static):
            e = self.gated_residual_network(
                embedding[:, i:i + 1, :],
                hidden_layer_size,
                dropout_rate=dropout_rate,
                use_time_distributed=False
            )
            trans_emb_list.append(e)

        transformed_embedding = concat(trans_emb_list, axis=1)

        combined = tf.keras.layers.Multiply(name="mult")([sparse_weights, transformed_embedding])

        static_vec = K.sum(combined, axis=1)

        return static_vec, sparse_weights
    
    def build_model(self):
        """
        Builds and compiles the model based on provided hyperparameters.
        
        Returns:
        --------
        model : tf.keras.Model
            The compiled model.
        """
        # Define Input Layers
        static_input = tf.keras.layers.Input(shape=(self.hyperparameters["n_static_features"]))
        dynamic_input = tf.keras.layers.Input(shape=(self.hyperparameters["n_timesteps"], 
                                                     self.hyperparameters["n_dynamic_features"]))
        masked = tf.keras.layers.Masking(mask_value=self.hyperparameters["mask_value"])(dynamic_input)

        # Get the embeddings
        static_emb = self.get_TFT_embeddings_no_categorical(static_input, 
                                                    self.hyperparameters["n_static_features"],
                                                    self.hyperparameters["layers_static"])

        # Perform the FS of the static features and pass through a GRN
        static_encoder, static_weights = self.static_combine_and_mask(static_emb, 
                                                                    self.hyperparameters["layers_static"], 
                                                                    self.hyperparameters["dropout_rate"])
        static_context_h = self.gated_residual_network(static_encoder,
                                                        self.hyperparameters["layers_dynamic"],
                                                        output_size=self.hyperparameters["layers_dynamic"],
                                                        dropout_rate=self.hyperparameters["dropout_rate"],
                                                        use_time_distributed=False)


        lstm_encoder = tf.keras.layers.GRU(
            self.hyperparameters["layers_dynamic"],
            dropout=self.hyperparameters["dropout_rate"],
            return_sequences=False,
            activation='tanh',
            use_bias=True
        )(masked, initial_state=[static_context_h])
        
        output = tf.keras.layers.Dense(1, activation="sigmoid")(lstm_encoder)


        # Ensure custom_losses.weighted_binary_crossentropy is defined
        customized_loss = custom_losses.weighted_binary_crossentropy(self.hyperparameters)

        # Compile the Model
        myOptimizer = tf.keras.optimizers.Adam(learning_rate=self.hyperparameters["learning_rate"])
        model = tf.keras.Model([static_input, dynamic_input], [output])
        model.compile(loss=customized_loss, optimizer=myOptimizer)
        
        return model

# Example Usage:
# hyperparameters = { 'n_static_features': ... , 'n_timesteps': ..., 'n_dynamic_features': ..., ... }
# fhsi_model = FHSI(hyperparameters)
# model = fhsi_model.build_model()
