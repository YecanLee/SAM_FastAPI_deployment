from fastapi import FastAPI, UploadFile, HTTPException, Body
from fastapi.responses import StreamingResponse
from ultralytics import SAM, YOLO
from matplotlib import pyplot as plt
from PIL import Image
import io
import uvicorn
from enum import Enum
from PIL import Image
import torch
from transformers import SamModel, SamProcessor, SamImageProcessor
from pydantic import BaseModel
from PIL import Image

# This model needs to be imported from the notebooks folder, this is probably a weird bug but I havn't figured out how to fix it yet
from visualization_helper import (show_points_and_boxes_on_image, show_points_on_image, show_boxes_on_image, show_mask, show_points, show_box, visualize_segmentation)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## DEBUG REMEBER TO CHANGE THE PATH FILE INTO A RELATIVE ONE!
# an utility function to load the model
def load_model(model_name: str):
    if model_name == 'sam':
        return SAM(model='sam_b.pt')
    elif model_name == 'yolov8':
        return YOLO(model='yolov8n.pt')
    elif model_name == 'sam_hf':
        return SamModel.from_pretrained('facebook/sam-vit-huge').to(device)
    else:
        raise HTTPException(status_code=400, detail="Model not found LOSER")
    

# pack the SAM model into a function
def perform_segmentation(image, input_points):
    """
    The following function is defined based on the SAM official documentation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    inputs = processor(image, return_tensors="pt").to(device)
    image_embeddings = model.get_image_embeddings(inputs["pixel_values"])

    inputs = processor(image, input_points=input_points, return_tensors="pt").to(device)
    # pop the pixel_values as they are not needed
    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped"])

    scores = outputs.iou_scores
    return masks, scores


def perform_sam(image, input_points):
    masks, scores = perform_segmentation(image, input_points)
    byte_io = visualize_segmentation(image, masks[0], scores[0])
    return byte_io


def perform_none_sam(image, model_name):
    model = load_model(model_name)
    results = model(image)
    im = Image.fromarray(results[..., ::-1])
    im.show()
    # Convert the image array to a byte stream
    byte_io = io.BytesIO()
    im.save(byte_io, format='JPEG')

    # Seek to the beginning of the byte stream
    byte_io.seek(0)

    return byte_io

model_name_dictionaly = {
    "sam": perform_none_sam,
    "yolov8": perform_none_sam,
    # "transunet": perform_none_sam,
    "sam_hf": perform_sam,
}


class Model_Name(str, Enum):
    sam = 'sam'
    yolov8 = "yolov8"
    transunet = "transunet"
    transunet_f = "linear_transunet"
    sam_hf = "sam_hf"


# DEBUG list[list]
class SegmentRequest(BaseModel):
    """
    The result should be a list of points, each point is a list of two numbers,
    representing the x and y coordinates of the point.
    """
    input_points: list[float]


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


# visualize the input image first 
@app.post("/image")
async def image_endpoint(file: UploadFile):
    """
    The user will first see their uploaded image before getting processed.
    In case they chose the wrong image, they can go back and choose the right one.
    """
    # Load the file into the memory
    image_data = await file.read()
    raw_image = Image.open(io.BytesIO(image_data)).convert('RGB')

    # Visualize the input
    plt.imshow(raw_image)

    # Convert the image array to a byte stream
    byte_io = io.BytesIO()
    raw_image.save(byte_io, format='JPEG')

    # Seek to the beginning of the byte stream
    byte_io.seek(0)

    return StreamingResponse(byte_io, media_type="image/jpeg")


# Build a app post method for choosing the model to use for different tasks
# using self-selected models besides the SAM offered by huggingface
@app.post("/model/{model_name}")
async def model_endpoint(model_name:Model_Name, file: UploadFile, segment_request:SegmentRequest = Body(...)):
    """
    The user can upload the images they want to use for the task they would like to use,
    and get the result directly by using our application.
    """
    # Load the file into the memory
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    handler = model_name_dictionaly.get(model_name.value)
    if handler is None:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model_name}")
    if model_name == "sam_hf":
        byte_io = handler(image, segment_request.input_points)
    else:
        byte_io = handler(image, model_name)
    return StreamingResponse(byte_io, media_type="image/jpeg")

### TO DO CHANGE THE APP POST, BUILD A NEW APP POST FOR THE SAM MODEL


if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8000)
