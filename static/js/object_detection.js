$('#objectDetectionModal').on('shown.bs.modal', function () {
    document.getElementById('video-stream-object-detection').src = "{{ url_for('object_detection_feed') }}";
});

$('#objectDetectionModal').on('hidden.bs.modal', function () {
    document.getElementById('video-stream-object-detection').src = "";
});

$('#faceRecognitionModal').on('shown.bs.modal', function () {
    document.getElementById('video-stream-face-recognition').src = "{{ url_for('face_recognition_feed') }}";
});

$('#faceRecognitionModal').on('hidden.bs.modal', function () {
    document.getElementById('video-stream-face-recognition').src = "";
});

$('#objectDetectionModal').on('shown.bs.modal', function () {
    document.getElementById('video-stream-object-detection').src = "{{ url_for('object_detection_feed') }}";
});

$('#objectDetectionModal').on('hidden.bs.modal', function () {
    document.getElementById('video-stream-object-detection').src = "";
});

$('#faceRecognitionModal').on('shown.bs.modal', function () {
    document.getElementById('video-stream-face-recognition').src = "{{ url_for('face_recognition_feed') }}";
});

$('#faceRecognitionModal').on('hidden.bs.modal', function () {
    document.getElementById('video-stream-face-recognition').src = "";
});

$('#cameraModal').on('shown.bs.modal', function (e) {
    document.getElementById("video-stream").src = "/video_feed";
});

$('#cameraModal').on('hidden.bs.modal', function (e) {
    document.getElementById("video-stream").src = "";
    
    document.getElementById("button-container").style.display = "block";
});

document.getElementById("try-now-button").addEventListener("click", function(event){
    event.preventDefault(); 

    fetch("/start_flask_app")
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                document.getElementById("video-stream").src = "/video_feed";
                document.getElementById("button-container").style.display = "none";
            } else {
                alert("Failed to start Flask application!");
            }
        })
        .catch(error => {
            console.error('There was an error with the fetch operation:', error);
        });
});