function createCanvases(batchSize, imageHeight, imageWidth, ...parentElement) {
    const canvases = [];

    for(let i = 0; i < batchSize; i++) {
        const element = document.createElement("canvas");

        element.id = "canvas-" + i;
        element.width = imageWidth;
        element.height = imageHeight;
        element.getContext("2d");

        if(parentElement.length === 1) {
            document.getElementById(parentElement[0]).appendChild(element);
        }
        else {
            document.body.appendChild(element);
        }

        canvases.push(element);
    }

    return canvases;
}

function displayTensor(pixelsTensor, canvas) {
    tf.browser.toPixels(pixelsTensor, canvas).then(() => {
        pixelsTensor.dispose();
    });
}

function displayTensorArray(tensorArray, canvasArray) {
    tensorArray.forEach((pixels, index) => {
        displayTensor(tf.tensor(pixels, [120, 80, 1], 'int32'), canvasArray[index]);
    });
}

const canvases = createCanvases(10, 120, 80, "predictions");

const socket = io('http://localhost:8080', {reconnectionDelay: 300, reconnectionDelayMax: 300});

socket.emit('startTraining');

socket.on('imageGenerated', (batchImageArray) => {
    displayTensorArray(batchImageArray, canvases);
});