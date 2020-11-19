var loadFile = function (event) {
  var image = document.getElementById("output");
  image.style.height = document.querySelector("#ht").value + "px";
  image.style.weight = document.querySelector("#wt").value + "px";

  image.src = URL.createObjectURL(event.target.files[0]);
};
