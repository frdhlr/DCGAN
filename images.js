const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');

function padImageNumber(imageNumber, imageNumberLength) {
    return imageNumber.toString().padStart(imageNumberLength, "0");
}

function createImageName(paddedImageNumber) {
    return `./Data/Bayeux_${paddedImageNumber}.jpg`;
}

function addNoiseToTensor(initialTensor) {
    const noisedTensor = tf.tidy(() => {
        return initialTensor.add(tf.randomNormal(initialTensor.shape, 0, 0.1));
    });

    return noisedTensor;
}

function normalizeTensor(nonNormalizedTensor) {
    const normalizedBatchTensor = tf.tidy(() => {
        return nonNormalizedTensor.add(-127.5).div(127.5);
    });

    return normalizedBatchTensor;
}

 function denormalizeTensor(normalizedTensor) {
     const denormalizedTensor = tf.tidy(() => {
         return normalizedTensor.mul(127.5).add(127.5).maximum(0);
     });

     return denormalizedTensor;
}

function loadNormalizedImageTensor(paddedImageNumber) {
    const image = fs.readFileSync(createImageName(paddedImageNumber));

    const normalizedImageTensor = tf.tidy(() => {
        return normalizeTensor(tf.node.decodeJpeg(image, 1));
    });

    return normalizedImageTensor;
}

function loadBatchTensor(nbImages, batchSize) {
    const imageNumberLength = nbImages.toString().length;
    const batchArray = [];

    const batchTensor = tf.tidy(() => {
        for(let i = 0; i < batchSize; i++) {
            const paddedImageNumber = padImageNumber(Math.floor(Math.random() * nbImages) + 1, imageNumberLength);

            batchArray.push(loadNormalizedImageTensor(paddedImageNumber).arraySync());
        }

        return tf.tensor4d(batchArray);
    })

    return batchTensor;
}

function generateImageArrayFromBatchTensor(normalizedbatchTensor) {
    imageArray = tf.tidy(() => {
        denormalizedBatchTensor = denormalizeTensor(normalizedbatchTensor);

        return denormalizedBatchTensor.arraySync();
    });

    return imageArray;
}

module.exports = {
    addNoiseToTensor,
    loadBatchTensor,
    generateImageArrayFromBatchTensor
}

