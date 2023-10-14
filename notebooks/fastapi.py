from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import SAM, YOLO
from PIL import Image
import io
import unicorn
from enum import Enum

class Model_Name(str, Enum):
    sam = 'Segment Anything Model'
    yolov8 = "Yolov8 unfinetuned model"
    transunet = "TransUNet model"
    transunet_f = "TransUNet model with flatten attention mechanism"

class Model_Built(str, Enum):
    sam = SAM(model = 'sam_b.pt')
    yolov8 = YOLO(model = 'yolov8n.pt')

# Initialize the building of the application
app = FastAPI()

@app.get('/')
async def main():
    """
    confirmation information of successful login
    """
    return "Thank you for using our application, your service has been successfully started"

@app.get("/model/{model_name}")
async def get_model_name(model_name:Model_Name):
    """
    The user can get the model name they want to use for the task they would like to perform.
    """
    if model_name is Model_Name.sam:
        return {'Model Name': model_name, "message":"Segmentation model "}
    if model_name is Model_Name.yolov8:
        return {'Model Name': model_name, "message":"Object detection model"}
    if model_name is Model_Name.transunet:
        return {'Model Name': model_name, "message":"Segmentation model "}
    if model_name is Model_Name.transunet_f:
        return {'Model Name': model_name, "message":"Segmentation model "}
    return {'Model Name': model_name, "message":"Model not found"}

# Build a app post method for choosing the model to use for different tasks
@app.post("/model/{model_name}")
async def model_endpoint(model_name:Model_Name, file: UploadFile):
    """
    The user can upload the images they want to use for the task they would like to use,
    and get the result directly by using our application.
    """
    # Load the file into the memory
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Now you can use the 'model' object to run inference on the image
    results = Model_Built[model_name](image)

    # Show the results
    for r in results:
        im_array = r.plot()

@app.post("/detect_objects")
async def segment_images_endpoint(file: UploadFile):
    """
    The user can upload the images they want to use for the segmentation task,
    and get the result directly by using our application.
    """
    # Load the file into the memory
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Now you can use the 'model' object to run inference on the image
    results = model(image)

    # Show the results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image
        im.save('results.jpg')  # save image

    # Convert the image array to a byte stream
    byte_io = io.BytesIO()
    im.save(byte_io, format='JPEG')

    # Seek to the beginning of the byte stream
    byte_io.seek(0)

    return StreamingResponse(byte_io, media_type="image/jpeg")


if __name__ == "__main__":
    unicorn.run(app, host = "0.0.0.0", port = 8000)
