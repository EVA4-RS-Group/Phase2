Title: Week4 Face Recognition
date: 2020-08-22
Javascripts: main.js

The task is to face recognition from 10 known celebrities - Aishwarya Rai, Elon Musk, Mahendra Singh Dhoni, Malala Yousafzai, Narendra Modi, Priyanka Chopra, Rahul Gandhi, Sachin Tendulkar, Shahrukh Khan, Shreya Ghoshal.

  <section>
    <div class="row gtr-uniform">
      <div class="col-3 col-12-xsmall">
        <ul class="actions">
          <li><input id="getFile" type="file" accept="image/jpg" name="files[]" multiple/></li>
        </ul>
        <ul class="actions">
          <li><input id="faceRecog" type="button" value="Face Recognition"/></li>
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
