Put a SQUARE, RELATIVELY LOW RESOLUTION (max 256x256), PNG, image into your root folder (I like to put it in a 'data' subfolder and use '/data/image.png')

Create an environment using `python -m venv .venv`
Start your venv using `.\.venv\Scripts\activate`
Install the dependencies using `pip install -r requirements.txt`

Then run the program using `python shape-painting.py <img_path="data/myimage.png">`
