import tensorflow as tf

# 1. Setup Data Paths and Parameters
# Make sure your 'Skin_Data' folder is in the same directory as this script, 
# with 'Infected' and 'Normal' subfolders inside.
DATA_DIR = "Skin_Data" 
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

print("Loading your skin images...")
# Load Training Data
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, 
    validation_split=0.2, 
    subset="training", 
    seed=123, 
    image_size=IMG_SIZE, 
    batch_size=BATCH_SIZE
)

# Load Validation Data
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, 
    validation_split=0.2, 
    subset="validation", 
    seed=123, 
    image_size=IMG_SIZE, 
    batch_size=BATCH_SIZE
)

print("Building the AI Brain...")
# 2. Build the Model using MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), 
    include_top=False, 
    weights='imagenet'
)
base_model.trainable = False # Freeze the pre-trained weights so we don't break them

# 3. Stack the layers together
model = tf.keras.Sequential([
    # THE FIX: This scales the raw image pixels (0 to 255) down to (-1 to 1)
    tf.keras.layers.Rescaling(1./127.5, offset=-1), 
    
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid') # Outputs a probability between 0 and 1
])

# 4. Compile the Model
model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

print("Training the AI (This might take a few minutes)...")
# 5. Train the Model
model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=10 # Increased to 10 so the AI studies the images longer
)

# 6. Save the resulting brain
model.save('skin_model.keras')
print("SUCCESS! Model trained and saved as 'skin_model.keras'!")