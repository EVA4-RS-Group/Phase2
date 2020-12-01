Title: Week-13 Speech Recognition
date: 2020-12-01
Javascripts: main.js


Detecting Speech commands like ['cat', 'dog', 'six', 'bird', 'eight', 'no', 'tree', 'marvin','left','down', 'off', 'on', 'five', 'three', 'go', 'seven', 'sheila', 'right', 'four', 'happy', 'bed', 'zero', 'one', 'wow', 'two', 'yes','house', 'up', 'nine', 'stop']


  <section>
    <div class="row gtr-uniform">
      <div class="col-3 col-12-xsmall">
        <ul class="actions">
          <li><input id="getFile" type="file" accept="audio/wav"/></li>
        </ul>
        <ul class="actions">
          <li><input id="srecog" type="button" value="SpeechRecog"/></li>
        </ul>
      </div>
      <div class="col-6 col-12-xsmall">
        <span class="image fit">
          <img id="upImage" src="https://icons.iconarchive.com/icons/alecive/flatwoken/512/Apps-Player-Audio-icon.png" alt="">
        </span>
        <h3 id="imgClass" style="text-align:center" ></h3>
      </div>
    </div>
  </section>
