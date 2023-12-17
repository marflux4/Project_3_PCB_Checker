from ultralytics import YOLO

# TRAINING THE MODEL ON THE DATA FILE PROVIDED
# Load yolov8 nano
# model = YOLO("yolov8n.pt") # build a new model from scratch

# Train model using specific hyperparameters
# results = model.train(data="datasets/data.yaml", epochs=50, batch=4, imgsz=928, name='Project_3_Model_Martin3')


# EVALUATING 3 IMAGES ON TRAINED MODEL
# Load the trained weights for evaluation
model = YOLO("Project_3_Model_Martin3/weights/best.pt")  # Update the path accordingly

results = model.predict(['datasets/evaluation/ardmega.jpg', 'datasets/evaluation/arduno.jpg', 'datasets/evaluation/rasppi.jpg'], save=True, imgsz=900)

# The 3 evaluated images can be found in runs>detect>predict folder 