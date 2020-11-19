from django.shortcuts import render
from django.views import View

from PIL import Image
from io import BytesIO

import cloudinary.uploader as uploader


from .algo import carve_main

# Create your views here.


def carve(image, height, width):

    img = Image.open(BytesIO(image))
    name = "temp.jpeg"
    out = "out.jpeg"

    with open(name, "wb") as fp:
        pk = img.save(fp)

    old = uploader.upload_resource(BytesIO(image), folder="SeamCarving")



    carve_main(name, out, height, width)

    with open(out, "rb") as fp:
        pk = fp.read()

    image = uploader.upload_resource(pk, folder="SeamCarving")

    return (old.url, image.url)


class HomeView(View):

    def get(self, request, *args, **kwargs):
        return render(request, "index.html", {})

    def post(self, request, *args, **kwargs):
        image = request.FILES['image'].file.read()
        height = int(request.POST['height'])
        width = int(request.POST['width'])

        old, new = carve(image, height, width)

        return render(request, "index.html", {"old": old, "new": new})








