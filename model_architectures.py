import tensorflow as tf

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

# Layer utility functions.
def linear_layer(size,
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

def add_and_norm(x_list):
    """Applies skip connection followed by layer normalisation.

  Args:
    x_list: List of inputs to sum for skip connection

  Returns:
    Tensor output from layer.
  """
    tmp = Add()(x_list)
    tmp = LayerNorm()(tmp)
    return tmp

def apply_gating_layer(x,
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



def gated_residual_network(x,
                           hidden_layer_size,
                           output_size=None,
                           dropout_rate=None,
                           use_time_distributed=True,
                           additional_context=None,
                           return_gate=False):
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
    hidden = linear_layer(
        hidden_layer_size,
        activation=None,
        use_time_distributed=use_time_distributed)(
        x)
    if additional_context is not None:
        hidden = hidden + linear_layer(
            hidden_layer_size,
            activation=None,
            use_time_distributed=use_time_distributed,
            use_bias=False)(
            additional_context)
    hidden = tf.keras.layers.Activation('elu')(hidden)
    hidden = linear_layer(
        hidden_layer_size,
        activation=None,
        use_time_distributed=use_time_distributed)(
        hidden)

    gating_layer, gate = apply_gating_layer(
        hidden,
        output_size,
        dropout_rate=dropout_rate,
        use_time_distributed=use_time_distributed,
        activation=None)

    if return_gate:
        return add_and_norm([skip, gating_layer]), gate
    else:
        return add_and_norm([skip, gating_layer])





def get_static_embeddings(static_input,
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


def static_combine_and_mask(embedding,
                            hidden_layer_size,
                            dropout_rate=None
                            ):
    # Flatten the static features
    _, num_static, _ = embedding.get_shape().as_list()
    flatten = tf.keras.layers.Flatten(name="flattened_inputs")(embedding)

    # Create the weights of the static features passing the input through a GRN, and then a softmax.
    mlp_outputs = gated_residual_network(
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
        e = gated_residual_network(
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
