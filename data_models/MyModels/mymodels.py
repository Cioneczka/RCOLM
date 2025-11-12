
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
#             layers.Conv2D(64, (3, 3), activation='relu', padding='same', kerner_initializer='he_normal'),
#             layers.BatchNormalization(),
#             layers.MaxPooling2D((2, 2)),
 

            layers.Flatten() 
            layers.Dropout(0.35)
            layers.Dense(64, activation='relu)
            layers.Dense(num_classes, activation='softmax'),  # głowa klasyfikacyjna

        ])
         

        return model




