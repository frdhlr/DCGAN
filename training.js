const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');

const im = require('./images');

function getDiscriminatorTrueTrainSet(nbImages, trainingBatchSize) {
    const discriminatorTrueTrainSet = tf.tidy(() => {
        const initialX = im.loadBatchTensor(nbImages, trainingBatchSize);
        const XTrue = im.constrainTensor(im.addNoiseToTensor(initialX));
        const yTrue = tf.randomUniform([trainingBatchSize, 1], 0.0, 0.2);

        return {X: XTrue, y: yTrue};
    });

    return discriminatorTrueTrainSet;
}

function getDiscriminatorFalseTrainSet(generator, trainingBatchSize, inputDim) {
    const discriminatorFalseTrainSet = tf.tidy(() => {
        const initialX = generator.predict(tf.randomNormal([trainingBatchSize, inputDim], 0, 1));
        const XFalse = im.constrainTensor(im.addNoiseToTensor(initialX));
        const yFalse = tf.randomUniform([trainingBatchSize, 1], 0.8, 1.0);

        return {X: XFalse, y: yFalse};
    });

    return discriminatorFalseTrainSet;
}

function getGeneratorTrainSet(trainingBatchSize, inputDim) {
    const generatorTrainSet = tf.tidy(() => {
        const XInput = tf.randomNormal([trainingBatchSize, inputDim], 0, 1);
        const yTrue  = tf.zeros([trainingBatchSize, 1]);

        return {X: XInput, y: yTrue};
    })

    return generatorTrainSet;
}

function freeTrainSetTensors(trainSet) {
    trainSet.X.dispose();
    trainSet.y.dispose();
}

async function trainModels(generator, discriminator, adversarial, nbEpochs, nbImages, trainingBatchSize, inputDim) {
    const discriminatorTrueTrainSet = getDiscriminatorTrueTrainSet(nbImages, trainingBatchSize);
    const discriminatorTrueResult  = await discriminator.fit(discriminatorTrueTrainSet.X, discriminatorTrueTrainSet.y);
    freeTrainSetTensors(discriminatorTrueTrainSet);

    const discriminatorFalseTrainSet = getDiscriminatorFalseTrainSet(generator, trainingBatchSize, inputDim);
    const discriminatorFalseResult = await discriminator.fit(discriminatorFalseTrainSet.X, discriminatorFalseTrainSet.y);
    freeTrainSetTensors(discriminatorFalseTrainSet);

    discriminator.trainable = false;

    const generatorTrainSet = getGeneratorTrainSet(trainingBatchSize, inputDim);
    const generatorResult = await adversarial.fit(generatorTrainSet.X, generatorTrainSet.y);
    freeTrainSetTensors(generatorTrainSet);

    discriminator.trainable = true;
}

function generateImage(generator, generatingBatchSize, inputDim) {
    const imageArray = tf.tidy(() => {
        const input = tf.randomNormal([generatingBatchSize, inputDim], 0, 1);
        const batchTensor = generator.predict(input);

        return im.generateImageArrayFromBatchTensor(batchTensor);
    });

    return imageArray;
}

function getPendingEpoch(handlingPath) {
    let epoch = 1;

    fs.readdirSync(`${handlingPath}`).forEach((fileName) => {
        if(fileName.includes('model_')) {
            const startindex  = fileName.indexOf('_') + 1;
            const endIndex    = fileName.indexOf('.');

            epoch = parseInt(fileName.substring(startindex, endIndex), 10) + 1;
        }
    });

    return epoch;
}

module.exports = {
    trainModels,
    generateImage,
    getPendingEpoch
}