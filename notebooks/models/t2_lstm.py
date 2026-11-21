from tensorflow.keras import Model
from tensorflow.keras.layers import (
    LSTM, Concatenate, Dense, Input, MultiHeadAttention, RepeatVector,
)

from notebooks.t2_preprocessing import N_PRED


def build_model(input_shape, use_attention=True):
    inputs = Input(shape=input_shape)
    encoder_sequence, encoder_h, encoder_c = LSTM(
        64, return_sequences=True, return_state=True, name="encoder"
    )(inputs)

    decoder_inputs = RepeatVector(N_PRED)(encoder_h)
    decoder_sequence = LSTM(64, return_sequences=True, name="decoder")(
        decoder_inputs, initial_state=[encoder_h, encoder_c]
    )

    features = decoder_sequence
    if use_attention:
        context = MultiHeadAttention(num_heads=4, key_dim=16, name="cross_attention")(
            query=decoder_sequence, key=encoder_sequence, value=encoder_sequence
        )
        features = Concatenate()([decoder_sequence, context])

    features = Dense(20, activation="relu", kernel_initializer="he_normal")(features)
    outputs = Dense(1, name="consumption")(features)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model