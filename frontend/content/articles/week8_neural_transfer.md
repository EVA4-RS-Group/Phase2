Title: Week8 Neural Style Transfer
date: 2020-10-11
Javascripts: main.js

The task is to upload a src and dest image and neural style transfer the dest image with src image. Please note you will have to upload two images

  <section>
    <div class="row gtr-uniform">
      <div class="col-3 col-12-xsmall">
        <ul class="actions">
          <li><input id="getFile" type="file" accept="image/jpg" name="files[]" multiple/></li>
        </ul>
        <ul class="actions">
          <li><input id="styleTransfer" type="button" value="Style Transfer"/></li>
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
