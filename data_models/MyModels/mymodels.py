

from tensorflow.keras import layers, models

class MyModels:
    
    @staticmethod
    def CNN_model(num_classes):
        model = models.Sequential([
            layers.Input(shape=(128, 128, 3)),

            layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dropout(0.35),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax'),
        ])
        return model

        # ---------- UNET ----------

    @staticmethod
    def unet_conv_block(x, ch, k=(3, 3)):
        x = layers.Conv2D(ch, k, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(ch, k, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    @staticmethod
    def build_unet(input_shape, num_bins=1):
        inp = layers.Input(shape=input_shape)

        # ------- Encoder (3 poziomy) -------
        c1 = MyModels.unet_conv_block(inp, 32)      # 360 x 216
        p1 = layers.MaxPooling2D((2, 2))(c1)        # 180 x 108

        c2 = MyModels.unet_conv_block(p1, 64)       # 180 x 108
        p2 = layers.MaxPooling2D((2, 2))(c2)        # 90 x 54

        c3 = MyModels.unet_conv_block(p2, 128)      # 90 x 54
        p3 = layers.MaxPooling2D((2, 2))(c3)        # 45 x 27

        # ------- Bottleneck -------
        b = MyModels.unet_conv_block(p3, 256)       # 45 x 27

        # ------- Decoder -------
        u3 = layers.UpSampling2D((2, 2))(b)         # 90 x 54
        u3 = layers.Concatenate()([u3, c3])
        u3 = MyModels.unet_conv_block(u3, 128)

        u2 = layers.UpSampling2D((2, 2))(u3)        # 180 x 108
        u2 = layers.Concatenate()([u2, c2])
        u2 = MyModels.unet_conv_block(u2, 64)

        u1 = layers.UpSampling2D((2, 2))(u2)        # 360 x 216
        u1 = layers.Concatenate()([u1, c1])
        u1 = MyModels.unet_conv_block(u1, 32)

        out = layers.Conv2D(num_bins, (1, 1), activation='sigmoid')(u1)

        return models.Model(inp, out, name='Unet_multiF0')

