
from tensorflow.keras import datasets, layers, models
class MyModels:
    @staticmethod
    def CNN_model(num_classes):
        model = models.Sequential([
            layers.Input(shape=(128, 128, 3)),

            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
#             layers.Conv2D(32, (3, 3), activation='relu'),
#             layers.BatchNormalization(),
#             layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0,1),

            layers.Dense(32, activation='relu'),
            layers.Dropout(0,1),

            layers.Dense(num_classes, activation='softmax'),  # głowa klasyfikacyjna
        ])

        return model




