from django.shortcuts import render
from django.views import View

from PIL import Image
from io import BytesIO

import cloudinary.uploader as uploader


from .algo import carve_main

# Create your views here.


def carve(image, height, width):

    pk = BytesIO(image)
    old = uploader.upload_resource(pk, folder="SeamCarving")
    pk.flush()

    new = carve_main(old.url, height, width)

    pk = BytesIO(new)
    image = uploader.upload_resource(pk, folder="SeamCarving")
    pk.flush()

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








