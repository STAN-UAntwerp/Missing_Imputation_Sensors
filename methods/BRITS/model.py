import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

class FeatureRegression(tf.keras.layers.Layer):
    
    def __init__(self):
        
        '''
        Feature regression layer, see Equation (7) in Section 4.3 of the BRITS paper.
        
        Parameters:
        __________________________________
        None.
        '''

        super(FeatureRegression, self).__init__()
        
    def build(self, input_shape):
        self.w = self.add_weight('w', shape=[int(input_shape[-1]), int(input_shape[-1])])
        self.b = self.add_weight('b', shape=[int(input_shape[-1])])
        self.d = tf.ones([input_shape[-1], input_shape[-1]]) - tf.eye(input_shape[-1], input_shape[-1])
        
    def call(self, inputs):
        
        '''
        Parameters:
        __________________________________
        inputs: tf.Tensor.
            Complement vector at a given time step, tensor with shape (samples, features)
            where samples is the batch size and features is the number of time series.

        Returns:
        __________________________________
        tf.Tensor.
            Feature-based estimation at a given time step, tensor with shape (samples, features)
            where samples is the batch size and features is the number of time series.
        '''
        
        return tf.matmul(inputs, self.w * self.d) + self.b


class TemporalDecay(tf.keras.layers.Layer):
    
    def __init__(self, units):
        
        '''
        Temporal decay layer, see Equation (3) in Section 4.1.1 of the BRITS paper.

        Parameters:
        __________________________________
        units: int.
            Number of hidden units of the recurrent layer.
        '''
        
        self.units = units
        super(TemporalDecay, self).__init__()
        
    def build(self, input_shape):
        self.w = self.add_weight('w', shape=[int(input_shape[-1]), self.units])
        self.b = self.add_weight('b', shape=[self.units])
        
    def call(self, inputs):
        
        '''
        Parameters:
        __________________________________
        inputs: tf.Tensor.
            Time gaps at a given time step, tensor with shape (samples, features) where samples
            is the batch size and features is the number of time series.

        Returns:
        __________________________________
        tf.Tensor.
            Temporal decay at a given time step, tensor with shape (samples, units) where samples
            is the batch size and units is the number of hidden units of the recurrent layer.
        '''
        
        return tf.exp(- tf.nn.relu(tf.matmul(inputs, self.w) + self.b))


class RITS(tf.keras.layers.Layer):
    
    def __init__(self, units):
        
        '''
        RITS layer, see Section 4.3 of the BRITS paper.

        Parameters:
        __________________________________
        units: int.
            Number of hidden units of the recurrent layer.
        '''
        
        self.units = units
        self.rnn_cell = None
        self.temp_decay = None
        self.hist_reg = None
        self.feat_reg = None
        self.weight_combine = None
        super(RITS, self).__init__()
        
    def build(self, input_shape):
        
        if self.rnn_cell is None:
            self.rnn_cell = tf.keras.layers.LSTMCell(units=self.units)

        if self.temp_decay is None:
            self.temp_decay = TemporalDecay(units=self.units)
        
        if self.hist_reg is None:
            self.hist_reg = tf.keras.layers.Dense(units=input_shape[2])

        if self.feat_reg is None:
            self.feat_reg = FeatureRegression()

        if self.weight_combine is None:
            self.weight_combine = tf.keras.layers.Dense(units=input_shape[2], activation='sigmoid')
        
    def call(self, inputs):
        
        '''
        Parameters:
        __________________________________
        inputs: tf.Tensor.
            Model inputs (time series, masking vectors and time gaps), tensor with shape (samples, timesteps, features, 3)
            where samples is the batch size, timesteps is the number of time steps, features is the number of time series
            and 3 is the number of model inputs.

        Returns:
        __________________________________
        outputs: tf.Tensor.
            Model outputs (imputations), tensor with shape (samples, timesteps, features) where samples is the batch size,
            timesteps is the number of time steps and features is the number of time series.
        
        loss: tf.Tensor.
            Loss value (sum of mean absolute errors), scalar tensor.
        '''
        
        # Get the inputs (time series, masking vectors and time gaps).
        values = tf.cast(inputs[:, :, :, 0], dtype=tf.float32)
        masks = tf.cast(inputs[:, :, :, 1], dtype=tf.float32)
        deltas = tf.cast(inputs[:, :, :, 2], dtype=tf.float32)

        # Initialize the outputs (imputations).
        outputs = tf.TensorArray(
            element_shape=(inputs.shape[0], inputs.shape[2]),
            size=inputs.shape[1],
            dynamic_size=False,
            dtype=tf.float32
        )
        
        # Initialize the states (memory state and carry state).
        h = tf.zeros(shape=(tf.shape(inputs)[0], self.units))
        c = tf.zeros(shape=(tf.shape(inputs)[0], self.units))
        
        # Initialize the loss (mean absolute error).
        loss = 0.
        
        # Loop across the time steps.
        for t in tf.range(inputs.shape[1]):
            
            # Extract the inputs for the considered time step.
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]
            
            # Run the history-based estimation, see Equation (1) in Section 4.1.1 of the BRITS paper.
            x_h = self.hist_reg(h)

            # Derive the complement vector, see Equation (2) in Section 4.1.1 of the BRITS paper.
            x_c = m * x + (1 - m) * x_h

            # Derive the temporal decay, see Equation (3) in Section 4.1.1 of the BRITS paper.
            gamma = self.temp_decay(d)
            
            # Run the feature-based estimation, see Equation (7) in Section 4.3 of the BRITS paper.
            z_h = self.feat_reg(x_c)
            
            # Derive the weights of the history-based and feature-based estimation, see Equation (8) in Section 4.3 of the BRITS paper.
            beta = self.weight_combine(tf.concat([gamma, m], axis=-1))

            # Combine the history-based and feature-based estimation, see Equation (9) in Section 4.3 of the BRITS paper.
            c_h = beta * z_h + (1 - beta) * x_h

            # Update the loss, see the last equation in Section 4.3 of the BRITS paper.
            loss += tf.reduce_sum((tf.abs(x - x_h) * m) / (tf.reduce_sum(m) + 1e-5))
            loss += tf.reduce_sum((tf.abs(x - z_h) * m) / (tf.reduce_sum(m) + 1e-5))
            loss += tf.reduce_sum((tf.abs(x - c_h) * m) / (tf.reduce_sum(m) + 1e-5))
            
            # Update the outputs, see Equation (10) in Section 4.3 of the BRITS paper.
            c_c = m * x + (1 - m) * c_h
            outputs = outputs.write(index=t, value=c_c)
            
            # Update the states, see Equation (11) in Section 4.3 of the BRITS paper.
            h, [h, c] = self.rnn_cell(states=[h * gamma, c], inputs=tf.concat([c_c, m], axis=-1))

        # Reshape the outputs.
        outputs = tf.transpose(outputs.stack(), [1, 0, 2])

        # Average the loss.
        loss /= (3 * inputs.shape[1])
        
        return outputs, loss

def get_inputs(x):
    
    '''
    Derive the masking vectors and calculate the time gaps.
    See Section 3 of the BRITS paper.
    
    Parameters:
    __________________________________
    x: np.array.
        Time series, array with shape (samples, features) where samples is the length of the time series
        and features is the number of time series.

    Returns:
    __________________________________
    inputs: tf.Tensor.
        Model inputs, tensor with shape (samples, features, 3) where 3 is the number of model inputs
        (time series, masking vectors and time gaps).
    '''
    
    # Derive the masking vector.
    m = np.where(np.isnan(x), 0, 1)
    
    # Calculate the time gaps.
    d = np.zeros(x.shape)
    for t in range(1, x.shape[0]):
        d[t, :] = np.where(m[t - 1, :] == 0, d[t - 1, :] + 1, 1)

    # Standardize the time gaps.
    d = (d - d.mean(axis=0)) / (d.std(axis=0) + 1e-5)

    # Mask the inputs.
    x = np.where(m == 0, 0, x)
    
    # Cast the inputs to float tensors.
    x = tf.expand_dims(tf.cast(x, tf.float32), axis=-1)
    m = tf.expand_dims(tf.cast(m, tf.float32), axis=-1)
    d = tf.expand_dims(tf.cast(d, tf.float32), axis=-1)

    # Concatenate the inputs.
    inputs = tf.concat([x, m, d], axis=-1)
    
    return inputs

class BRITS:
    
    def __init__(self, x, units, timesteps):
        
        '''
        Implementation of multivariate time series imputation model introduced in Cao, W., Wang, D., Li, J.,
        Zhou, H., Li, L. and Li, Y., 2018. BRITS: Bidirectional recurrent imputation for time series.
        Advances in neural information processing systems, 31.
        
        Parameters:
        __________________________________
        x: np.array.
            Time series, array with shape (samples, features) where samples is the length of the time series
            and features is the number of time series.
        
        units: int.
            Number of hidden units of the recurrent layer.

        timesteps: int.
            Number of time steps.
        '''
        
        self.x = x
        self.x_min = np.nanmin(x, axis=0)
        self.x_max = np.nanmax(x, axis=0)
        self.samples = x.shape[0]
        self.features = x.shape[1]
        self.units = units
        self.timesteps = timesteps
  
    def fit(self, learning_rate=0.001, batch_size=32, epochs=100, verbose=False):
        
        '''
        Train the model.
        
        Parameters:
        __________________________________
        learning_rate: float.
            Learning rate.
            
        batch_size: int.
            Batch size.
            
        epochs: int.
            Number of epochs.
            
        verbose: bool.
            True if the training history should be printed in the console, False otherwise.
        '''
        
        # Scale the time series.
        x = (self.x - self.x_min) / (self.x_max - self.x_min)

        # Get the inputs in the forward direction.
        forward = get_inputs(x)

        # Get the inputs in the backward direction.
        backward = get_inputs(np.flip(x, axis=0))

        # Generate the input sequences.
        dataset = tf.keras.utils.timeseries_dataset_from_array(
            data=tf.concat([forward, backward], axis=-1),
            targets=None,
            sequence_length=self.timesteps,
            sequence_stride=self.timesteps,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Build the model.
        model = build_fn(
            timesteps=self.timesteps,
            features=self.features,
            units=self.units
        )
        
        # Define the training loop.
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        @tf.function
        def train_step(data):
            with tf.GradientTape() as tape:
                
                # Calculate the loss.
                _, loss = model(data)
            
            # Calculate the gradient.
            gradient = tape.gradient(loss, model.trainable_variables)
        
            # Update the weights.
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))
        
            return loss
        
        # Train the model.
        for epoch in range(epochs):
            for data in dataset:
                loss = train_step(data)
            if verbose:
                print('epoch: {}, loss: {:,.6f}'.format(1 + epoch, loss))

        # Save the model.
        self.model = model
    
    def impute(self, x):
        
        '''
        Impute the time series.
        
        Parameters:
        __________________________________
        x: np.array.
            Time series, array with shape (samples, features) where samples is the length of the time series
            and features is the number of time series.
            
        Returns:
        __________________________________
        imputed: pd.DataFrame.
            Data frame with imputed time series.
        '''
        
        if x.shape[1] != self.features:
            raise ValueError(f'Expected {self.features} features, found {x.shape[1]}.')
        
        else:
            
            # Scale the time series.
            x = (x - self.x_min) / (self.x_max - self.x_min)
    
            # Get the inputs in the forward direction.
            forward = get_inputs(x)
    
            # Get the inputs in the backward direction.
            backward = get_inputs(np.flip(x, axis=0))
    
            # Generate the input sequences.
            dataset = tf.keras.utils.timeseries_dataset_from_array(
                data=tf.concat([forward, backward], axis=-1),
                targets=None,
                sequence_length=self.timesteps,
                sequence_stride=self.timesteps,
                batch_size=1,
                shuffle=False
            )
            
            # Generate the imputations.
            imputed = tf.concat([self.model(data)[0] for data in dataset], axis=0).numpy()
            imputed = np.concatenate([imputed[i, :, :] for i in range(imputed.shape[0])], axis=0)
            imputed = self.x_min + (self.x_max - self.x_min) * imputed

            return imputed


def build_fn(timesteps, features, units):
    
    '''
    Build the model, see Section 4.2 of the BRITS paper.
    
    Parameters:
    __________________________________
    features: int.
        Number of time series.

    timesteps: int.
        Number of time steps.
    
    units: int.
        Number of hidden units of the recurrent layer.
    '''
    
    # Define the input layer, the model takes 3 inputs (time series, masking
    # vectors and time gaps) for each direction (forward and backward).
    inputs = tf.keras.layers.Input(shape=(timesteps, features, 6))
    
    # Get the imputations and loss in the forward directions.
    forward_imputations, forward_loss = RITS(units=units)(inputs[:, :, :, :3])

    # Get the imputations and loss in the backward directions.
    backward_imputations, backward_loss = RITS(units=units)(inputs[:, :, :, 3:])
    
    # Average the imputations across both directions (forward and backward).
    outputs = (forward_imputations + backward_imputations) / 2

    # Sum the losses (forward loss, backward loss and consistency loss).
    loss = forward_loss + backward_loss + tf.reduce_mean(tf.abs(forward_imputations - backward_imputations))
    
    return tf.keras.Model(inputs, (outputs, loss))
