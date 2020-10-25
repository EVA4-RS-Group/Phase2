'use strict';

(function($) {

  $(document).ready(function () {
    // Check for FileReader API (HTML5) support.
    if (!window.FileReader) {
      alert('This browser does not support the FileReader API.');
    }
  });

  $.retryAjax = function (ajaxParams) {
    var errorCallback;
    ajaxParams.tryCount = (!ajaxParams.tryCount) ? 0 : ajaxParams.tryCount;
    ajaxParams.retryLimit = (!ajaxParams.retryLimit) ? 2 : ajaxParams.retryLimit;
    ajaxParams.suppressErrors = true;

    if (ajaxParams.error) {
        errorCallback = ajaxParams.error;
        delete ajaxParams.error;
    } else {
        errorCallback = function () {

        };
    }

    ajaxParams.complete = function (jqXHR, textStatus) {
        if ($.inArray(textStatus, ['timeout', 'abort', 'error']) > -1) {
            this.tryCount++;
            if (this.tryCount <= this.retryLimit) {

                // fire error handling on the last try
                if (this.tryCount === this.retryLimit) {
                    this.error = errorCallback;
                    delete this.suppressErrors;
                }

                //try again
                console.log("Trying again");
                $.ajax(this);
                return true;
            }

            window.alert('There was a server error.  Please refresh the page.  If the issue persists, give us a call. Thanks!');
            return true;
        }
    };

    $.ajax(ajaxParams);
  };

  var url = {
    "week1": "https://jzworltk54.execute-api.ap-south-1.amazonaws.com/dev/classify",
    "week2": "https://3g8t28a24d.execute-api.ap-south-1.amazonaws.com/dev/classify",
    "face_swap": "https://gh1xz0gzpj.execute-api.ap-south-1.amazonaws.com/dev/face_swap",
    "face_mask": "https://gh1xz0gzpj.execute-api.ap-south-1.amazonaws.com/dev/face_mask",
    "face_align": "https://gh1xz0gzpj.execute-api.ap-south-1.amazonaws.com/dev/face_align",
    "face_recog": "https://qeqn72vc9d.execute-api.ap-south-1.amazonaws.com/dev/face_rec",
    "human_pose": "https://ppcgpv4fq8.execute-api.ap-south-1.amazonaws.com/dev/humanPose",
    "gan": "https://hc77zhfgcg.execute-api.ap-south-1.amazonaws.com/dev/GAN",
    "vae": "https://4p5t092168.execute-api.ap-south-1.amazonaws.com/dev/VAE",
    "srgan": "https://81vw72pqa6.execute-api.ap-south-1.amazonaws.com/dev/srGan",
<<<<<<< HEAD
    "sentiment": "https://81vw72pqa6.execute-api.ap-south-1.amazonaws.com/dev/sentiment",
=======
    "style_transfer": "https://gh1xz0gzpj.execute-api.ap-south-1.amazonaws.com/dev/face_swap",
>>>>>>> ba1baf52d09cac08e4fa489fc77b0bcb9f40e3e1
  };

  // Utils

  function getFileReader(imgId) {
    let reader = new FileReader();
    reader.onload = function(e) {
      $("#"+imgId).attr('src', e.target.result);
    };
    return reader;
  };
  var reader = getFileReader("upImage")

  function showImage(input) {
    if (input.files && input.files[0]) {
      $("#imgClass").text("");
      reader.readAsDataURL(input.files[0]);
    }
  }

  $("input#getFile").change(function(){
    showImage(this);
  });

  $("#classifyImage1").click(function(){
    return classify("week1")
  });

  $("#classifyImage2").click(function(){
    return classify("week2")
  });

  $("#faceRecog").click(function(){
    return classify("face_recog")
  });

  function classify(url_key) {
    var documentData = new FormData();

    // Post the file to url and get response
    documentData.append("body", $('input#getFile')[0].files[0]);
    $.retryAjax({
        url: url[url_key],
        type: 'POST',
        data: documentData,
        async: false,
        cache: false,
        contentType: false,
        processData: false,
        timeout:5000,
        success: function (response) {
            $("#imgClass").text(response.predicted)
        },
        error: function(e) {
          alert(e.statusText)
        }
    });
    return false;
  }

  $("#faceSwap").click(function(){
    var documentData = new FormData();
    $.each($('input#getFile')[0].files,function(i, file){
      documentData.append("files["+i+"]", file)
    });
    $.ajax({
      url: url["face_swap"],
      type: 'POST',
      data: documentData,
      async: false,
      cache: false,
      contentType: false,
      processData: false,
      timeout:5000,
      success: function (response) {
          $("#file0").attr('src', 'data:image/png;base64,'+ response["file0"][1])
      },
      error: function(e) {
        alert(e.statusText)
      }
    });
  })

  $("#styleTransfer").click(function(){
    var documentData = new FormData();
    $.each($('input#getFile')[0].files,function(i, file){
      documentData.append("files["+i+"]", file)
    });
    $.ajax({
      url: url["style_transfer"],
      type: 'POST',
      data: documentData,
      async: false,
      cache: false,
      contentType: false,
      processData: false,
      timeout:5000,
      success: function (response) {
          $("#file0").attr('src', 'data:image/png;base64,'+ response["file0"][1])
      },
      error: function(e) {
        alert(e.statusText)
      }
    });
  })

  $("#faceAlign").click(function(){
    var documentData = new FormData();
    $.each($('input#getFile')[0].files,function(i, file){
      documentData.append("files["+i+"]", file)
    });
    $.ajax({
      url: url["face_align"],
      type: 'POST',
      data: documentData,
      async: false,
      cache: false,
      contentType: false,
      processData: false,
      timeout:5000,
      success: function (response) {
          $("#file0").attr('src', 'data:image/png;base64,'+ response["file0"][1])
      },
      error: function(e) {
        alert(e.statusText)
      }
    });
  })

  $("#faceMask").click(function(){
    var documentData = new FormData();
    $.each($('input#getFile')[0].files,function(i, file){
      documentData.append("files["+i+"]", file)
    });
    $.ajax({
      url: url["face_mask"],
      type: 'POST',
      data: documentData,
      async: false,
      cache: false,
      contentType: false,
      processData: false,
      timeout:5000,
      success: function (response) {
          $("#file0").attr('src', 'data:image/png;base64,'+ response["file0"][1])
      },
      error: function(e) {
        alert(e.statusText)
      }
    });
  })

  $("#humanPose").click(function(){
    var documentData = new FormData();
    $.each($('input#getFile')[0].files,function(i, file){
      documentData.append("files["+i+"]", file)
    });
    $.ajax({
      url: url["human_pose"],
      type: 'POST',
      data: documentData,
      async: false,
      cache: false,
      contentType: false,
      processData: false,
      timeout:5000,
      success: function (response) {
          $("#file0").attr('src', 'data:image/png;base64,'+ response["file0"][1])
      },
      error: function(e) {
        alert(e.statusText)
      }
    });
  })

  $("#GAN").click(function(){

    $.ajax({
      url: url["gan"],
      type: 'GET',
      async: false,
      cache: false,
      contentType: false,
      processData: false,
      timeout:5000,
      success: function (response) {
          $("#file0").attr('src', 'data:image/png;base64,'+ response["file0"][1])
      },
      error: function(e) {
        alert(e.statusText)
      }
    });
  })

  $("#VAE").click(function(){

    $.ajax({
      url: url["vae"],
      type: 'GET',
      async: false,
      cache: false,
      contentType: false,
      processData: false,
      timeout:5000,
      success: function (response) {
          $("#file0").attr('src', 'data:image/png;base64,'+ response["file0"][1])
      },
      error: function(e) {
        alert(e.statusText)
      }
    });
  })

  $("#srGAN").click(function(){
    var documentData = new FormData();
    $.each($('input#getFile')[0].files,function(i, file){
      documentData.append("files["+i+"]", file)
    });
    $.ajax({
      url: url["srgan"],
      type: 'POST',
      data: documentData,
      async: false,
      cache: false,
      contentType: false,
      processData: false,
      timeout:5000,
      success: function (response) {
          $("#file0").attr('src', 'data:image/png;base64,'+ response["file0"][1])
      },
      error: function(e) {
        alert(e.statusText)
      }
    });
  })

  $("#sentimentalanalysis").click(function(){
    var documentData = new FormData();
    $.each($('input#getFile')[0].files,function(i, file){
      documentData.append("files["+i+"]", file)
    });
    $.ajax({
      url: url["sentiment"],
      type: 'POST',
      data: documentData,
      async: false,
      cache: false,
      contentType: false,
      processData: false,
      timeout:5000,
      success: function (response) {
          $("#file0").attr('src', 'data:image/png;base64,'+ response["file0"][1])
      },
      error: function(e) {
        alert(e.statusText)
      }
    });
  })
  // Display error messages.
  function onError(error) {
    alert(error.responseText);
  }

})(jQuery)
