import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

app = Flask(__name__)

# Set up the upload folder and allowed file extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the BLIP model and processor for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to generate captions using the BLIP model
def generate_caption(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # Generate the caption
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption

# Home route - upload images
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('file')
        filenames = []
        captions = []

        # Process each uploaded image
        for file in files:
            if file and allowed_file(file.filename):
                filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                file.save(filename)
                filenames.append(file.filename)

                # Generate caption for each image using the BLIP model
                caption = generate_caption(filename)
                captions.append(caption)

        # Zip filenames and captions together before passing to template
        files_with_captions = zip(filenames, captions)

        # After uploading and generating captions, render the result page
        return render_template('result.html', files_with_captions=files_with_captions)

    return render_template('index.html')  # Show upload form

if __name__ == '__main__':
    app.run(debug=True)
