const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');

function createGenerator(imageRows, imageCols, imageChannels, inputDim) {
    const depth   = 128;
    const rows    = Math.floor(imageRows / 4);
    const cols    = Math.floor(imageCols / 4);

    const initializer = tf.initializers.truncatedNormal({mean: 0, stddev: 0.02});

    const generator = tf.sequential();

    generator.add(tf.layers.dense({units: rows * cols * depth,
                                   activation: 'relu',
                                   inputDim: inputDim,
                                   kernelInitialize: initializer}));
    generator.add(tf.layers.reshape({targetShape: [cols, rows, depth]}));

    generator.add(tf.layers.upSampling2d({}));
    generator.add(tf.layers.conv2d({filters: 128,
                                    kernelSize: 3,
                                    padding: 'same',
                                    kernelInitialize: initializer}));
    generator.add(tf.layers.batchNormalization({momentum: 0.8,
                                                kernelInitialize: initializer}));
    generator.add(tf.layers.reLU());

    generator.add(tf.layers.upSampling2d({}));
    generator.add(tf.layers.conv2d({filters: 64,
                                    kernelSize: 3,
                                    padding: 'same',
                                    kernelInitialize: initializer}));
    generator.add(tf.layers.batchNormalization({momentum: 0.8,
                                                kernelInitialize: initializer}));
    generator.add(tf.layers.reLU());

    generator.add(tf.layers.conv2d({filters: imageChannels,
                                    kernelSize: 3,
                                    padding: 'same',
                                    kernelInitialize: initializer}));
    generator.add(tf.layers.activation({activation: 'tanh'}));

    generator.summary();

    return generator;
}

function createDiscriminator(imageRows, imageCols, imageChannels) {
    const dropout    = 0.25;
    const inputShape = [imageCols, imageRows, imageChannels];

    const initializer = tf.initializers.truncatedNormal({mean: 0, stddev: 0.02});

    const discriminator = tf.sequential();

    discriminator.add(tf.layers.conv2d({filters: 32,
                                        kernelSize: 3,
                                        strides: 2,
                                        inputShape: inputShape,
                                        padding: 'same',
                                        kernelInitialize: initializer}));
    discriminator.add(tf.layers.leakyReLU({alpha: 0.2}));
    discriminator.add(tf.layers.dropout({rate: dropout}));

    discriminator.add(tf.layers.conv2d({filters: 64,
                                        kernelSize: 3,
                                        strides: 2,
                                        padding: 'same',
                                        kernelInitialize: initializer}));
    discriminator.add(tf.layers.zeroPadding2d({padding: [[0, 1], [0,1]]}));
    discriminator.add(tf.layers.batchNormalization({momentum: 0.8,
                                                    kernelInitialize: initializer}));
    discriminator.add(tf.layers.leakyReLU({alpha: 0.2}));
    discriminator.add(tf.layers.dropout({rate: dropout}));

    discriminator.add(tf.layers.conv2d({filters: 128,
                                        kernelSize: 3,
                                        strides: 2,
                                        padding: 'same',
                                        kernelInitialize: initializer}));
    discriminator.add(tf.layers.batchNormalization({momentum: 0.8,
                                                    kernelInitialize: initializer}));
    discriminator.add(tf.layers.leakyReLU({alpha: 0.2}));
    discriminator.add(tf.layers.dropout({rate: dropout}));

    discriminator.add(tf.layers.conv2d({filters: 256,
                                        kernelSize: 3,
                                        strides: 2,
                                        padding: 'same',
                                        kernelInitialize: initializer}));
    discriminator.add(tf.layers.batchNormalization({momentum: 0.8}));
    discriminator.add(tf.layers.leakyReLU({alpha: 0.2}));
    discriminator.add(tf.layers.dropout({rate: dropout}));

    discriminator.add(tf.layers.flatten());
    discriminator.add(tf.layers.dense({units: 1,
                                       activation: 'sigmoid',
                                       kernelInitialize: initializer}));

    discriminator.summary();

    return discriminator;
}

function createAdversarial(generator, discriminator) {
    const adversarial = tf.sequential();

    adversarial.add(generator);

    discriminator.trainable = false;

    adversarial.add(discriminator);

    adversarial.summary();

    return adversarial;
}

function compilModels(generator, discriminator, adversarial) {
    adversarial.compile({optimizer: 'adam', loss: 'binaryCrossentropy'});

    discriminator.trainable = true;
    discriminator.compile({optimizer: 'adam', loss: 'binaryCrossentropy'});
}

async function saveModel(model, handlingPath, savingPath, epoch) {
    const oldName  = `${handlingPath}/model.json`;
    const newName  = `${handlingPath}/model_${epoch}.json`;

    fs.readdirSync(`${handlingPath}`).forEach((fileName) => {
        fs.unlinkSync(`${handlingPath}/${fileName}`);
    });

    return model.save(`${savingPath}`).then(() => {
        fs.renameSync(oldName, newName);
    })
}

async function loadModel(handlingPath, savingPath) {
    fs.readdirSync(`${handlingPath}`).forEach((fileName) => {
        if(fileName.includes('model_')) {
            const oldName  = `${handlingPath}/${fileName}`;
            const newName  = `${handlingPath}/model.json`;

            fs.renameSync(oldName, newName);
        }
    });

    return tf.loadLayersModel(`${savingPath}/model.json`);
}

module.exports = {
    createGenerator,
    createDiscriminator,
    createAdversarial,
    compilModels,
    saveModel,
    loadModel
}