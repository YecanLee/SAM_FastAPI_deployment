from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import SAM, YOLO
from PIL import Image
import io
import uvicorn
from enum import Enum
from PIL import Image


class Model_Name(str, Enum):
    sam = 'sam'
    yolov8 = "yolov8"
    transunet = "transunet"
    transunet_f = "linear_transunet"

# an utility function to load the model
def load_model(model_name: str):
    if model_name == 'sam':
        return SAM(model='sam_b.pt')
    elif model_name == 'yolov8':
        return YOLO(model='yolov8n.pt')
    else:
        raise HTTPException(status_code=400, detail="Model not found LOSER")

# Initialize the building of the application
app = FastAPI()

@app.get('/')
async def main():
    """
    confirmation information of successful login
    """
    return "Thank you for using our application, your service has been successfully started"

@app.get("/model/{model_name}")
async def get_model_name(model_name: Model_Name):
    """
    The user can get the model name they want to use for the task they would like to perform.
    """
    return {'Model Name': model_name}

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
    try:
        model = load_model(model_name.value)
    except HTTPException:
        return {"Name Error": "Model not found LOSER"}
    
    # Run model for different tasks
    results = model(image)

    # Visualize the results
    for r in results:
        result_image = r.plot()
        im = Image.fromarray(result_image[..., ::-1])
        im.show()
        im.save('results.jpg')

   
    # Convert the image array to a byte stream
    byte_io = io.BytesIO()
    im.save(byte_io, format='JPEG')

    # Seek to the beginning of the byte stream
    byte_io.seek(0)

    return StreamingResponse(byte_io, media_type="image/jpeg")


if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8000)
