
from tensorflow.keras import datasets, layers, models
class MyModels:
    @staticmethod
    def CNN_model(num_classes):
        model = models.Sequential([
            layers.Input(shape=(128, 128, 3)),

            layers.Conv2D(16, (3, 3), activation='relu', padding='same', kerner_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(32, (3, 3), activation='relu',padding='same', kerner_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
#             
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', kerner_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
 

            layers.Flatten(), 
            layers.Dropout(0.35),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax'),  # głowa klasyfikacyjna

        ])
       

        return model

    def Unet_conv_block(x, ch, k=(3,3)):
        x = layers.Conv2D(ch, k, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Relu()(x)
        x = layers.Conv2D(ch, k, padding='same')(x)
        x = layers.BatchNormalization()(x)
        return layers.Relu()(x)

    def build_unet(input_shape, num_bins):
        inp = layers.Input(shape=input_shape)
        #Encoder
        c1 = Unet_conv_block(inp, 32); p1 = layers.MaxPooling2D((2,2))(c1)
        c2 = Unet_conv_block(inp, 64); p2 = layers.MaxPooling2D((2,2))(c2)
        c3 = Unet_conv_block(inp, 128); p3 = layers.MaxPooling2D((2,2))(c3)
        c4 = Unet_conv_block(inp, 256); p4 = layers.MaxPooling2D((2,2))(c4)
        
        #Latent space 
        b = conv_bloack(p4, 364)

        #Decoder
        u4 = L.UpSampling2D((2,2))(b);   u4 = L.Concatenate()([u4, c4]);  u4 = conv_block(u4, 256)
        u3 = L.UpSampling2D((2,2))(u4);  u3 = L.Concatenate()([u3, c3]);  u3 = conv_block(u3, 128)
        u2 = L.UpSampling2D((2,2))(u3);  u2 = L.Concatenate()([u2, c2]);  u2 = conv_block(u2, 64)
        u1 = L.UpSampling2D((2,2))(u2);  u1 = L.Concatenate()([u1, c1]);  u1 = conv_block(u1, 32)

        out = L.Conv2D(1, (1,1), activation = 'sigmoid')(u1)
        return models.Model(inp, out, name='Unet_multiF0')

