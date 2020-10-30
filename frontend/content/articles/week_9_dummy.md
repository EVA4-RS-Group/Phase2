Title: Week9 Dummy
date: 2020-10-21
Javascripts: main.js

In this week we deployed our model inference code to S3. The network is mobile net v2 trained on custom drone dataset.


  <section>
    <div class="row gtr-uniform">
      <div class="col-3 col-12-xsmall">
        <ul class="actions">
          <li><input id="getFile" type="file" accept="image/jpg"/></li>
        </ul>
        <ul class="actions">
          <li><input id="classifyImage2" type="button" value="Classify"/></li>
        </ul>
      </div>
      <div class="col-6 col-12-xsmall">
        <span class="image fit">
          <img id="upImage" src="#" alt="">
        </span>
        <h3 id="imgClass" style="text-align:center" ></p>
      </div>
    </div>
  </section>
