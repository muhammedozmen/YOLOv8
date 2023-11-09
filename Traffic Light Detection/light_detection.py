from roboflow import Roboflow
rf = Roboflow(api_key="EqbvV0Yw6Og59OId4zZH")
project = rf.workspace().project("traffic-light-detection-0ocf2")
model = project.version(1).model

# infer on a local image
#print(model.predict("light1.jpg", confidence=40, overlap=30).json())

# visualize prediction
model.predict("light3.jpg", confidence=40, overlap=30).save("prediction.jpg")






