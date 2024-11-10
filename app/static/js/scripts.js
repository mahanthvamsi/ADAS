// app/static/js/scripts.js
document.getElementById('uploadForm').onsubmit = async function(event) {
    event.preventDefault();
    
    let formData = new FormData();
    let fileInput = document.getElementById('fileInput');
    formData.append('file', fileInput.files[0]);
    
    let response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });
    
    let result = await response.json();
    let resultText = "Objects Detected:\n";
    result.objects.forEach(obj => {
        resultText += `Class: ${obj.class}, Confidence: ${obj.confidence}, Box: ${obj.box}\n`;
    });
    document.getElementById('result').innerText = resultText;
};

console.log("Scripts loaded.");
