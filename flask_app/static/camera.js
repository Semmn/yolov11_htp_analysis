window.onload = function () {
    const video = document.getElementById('camera');
    if (video) {
      navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
        video.srcObject = stream;
      });
    }
  };
  
  function capturePhoto() {
    const video = document.getElementById('camera');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
  
    const photoData = canvas.toDataURL('image/jpeg');
  
    fetch('/upload_photo', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: 'photo=' + encodeURIComponent(photoData)
    }).then(() => {
      alert("📷 사진이 저장되었습니다.");
    });
  }
  
