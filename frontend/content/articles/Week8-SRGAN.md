Title: Week8 Super Resolution GAN
date: 2020-10-10
Javascripts: main.js

The task is to estimate human pose on a image using ONNX runtime.

  <section>
    <div class="row gtr-uniform">
      <div class="col-3 col-12-xsmall">
        <ul class="actions">
          <li><input id="getFile" type="file" accept="image/jpg" name="files"/></li>
        </ul>
        <ul class="actions">
          <li><input id="srGAN" type="button" value="Super Resolution"/></li>
        </ul>
      </div>
      <div class="col-6 col-12-xsmall">
        <span class="image fit">
          <img id="upImage" src="#" alt="">
        </span>
        <h3 id="imgClass" style="text-align:center" ></p>
      </div>
    </div>
    <div class="row gtr-uniform">
      <div class="col-6">
        <span class="image fit"><img id="file0" src="#" alt=""></span>
      </div>
    </div>
  </section>
