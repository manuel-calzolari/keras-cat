import numbers
import numpy as np
import six
from keras.layers import Input, Lambda, Embedding, Reshape, concatenate
from keras.engine import Model


def _eval_dim(embedding_dim, max_level):
    if isinstance(embedding_dim, six.string_types):
        if embedding_dim == 'sqrt':
            embedding_dim = max(1, int(np.sqrt(max_level + 1)))
        elif embedding_dim == 'log2':
            embedding_dim = max(1, int(np.log2(max_level + 1)))
        else:
            raise ValueError(
                'Invalid value for embedding_dim. Allowed string '
                'values are "sqrt" or "log2".')
    elif embedding_dim is None:
        embedding_dim = max_level + 1
    elif not isinstance(embedding_dim, (numbers.Integral, np.integer)):  # float
        if embedding_dim > 0.0:
            embedding_dim = max(1, int(embedding_dim * (max_level + 1)))
        else:
            embedding_dim = 0
    return embedding_dim


def InputCategorical(input_dim, categorical=None, max_level=None, embedding_dim='log2'):
    """Categorical features support with entity embeddings.

    # Arguments
        input_dim: int > 0. Size of the input.
        categorical: list of int. Interpreted as indices.
        max_level: list of int. Maximum integer level for every categorical feature.
        embedding_dim: int, float, string or list of them for every categorical feature (default='log2').
            Dimension of the dense embedding:
                - If int, then the value is the dimension.
                - If float, then the value is a percentage and
                  `int(embedding_dim * (max_level + 1))` is the dimension.
                - If "sqrt", then the value is `sqrt(max_level + 1)`.
                - If "log2", then the value is `log2(max_level + 1)`.

    # Example

        ```python
        model = Sequential()
        model.add(InputCategorical(input_dim=78,
                                   categorical=[0,
                                                30],
                                   max_level=[X[:, 0].max(),
                                              X[:, 30].max()]
                                   ))
        ```
    """
    len_categorical = len(categorical)
    if not isinstance(embedding_dim, list):
        embedding_dim = [embedding_dim] * len_categorical
    embedding_dim = [_eval_dim(embedding_dim[i], max_level[i]) for i in range(len_categorical)]
    inputs = Input(shape=(input_dim,))
    layers = []
    for i in range(input_dim):
        col = Lambda(lambda data: data[:, i:i+1], output_shape=(1,))(inputs)
        if i in categorical:
            col_embedding = Embedding(max_level[categorical.index(i)] + 1, embedding_dim[categorical.index(i)], input_length=1)(col)
            col_embedding_reshape = Reshape(target_shape=(embedding_dim[categorical.index(i)],))(col_embedding)
            layers.append(col_embedding_reshape)
        else:
            layers.append(col)
    outputs = concatenate(layers)
    model = Model(inputs=inputs, outputs=outputs)
    return model
