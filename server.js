const http     = require('http');
const socketio = require('socket.io');

const ct = require('./const');
const md = require('./models');
const tr = require('./training');

const port   = process.env.PORT || 8080;
const server = http.createServer();
const io     = socketio(server);

server.listen(port, () => {
    console.log('');
    console.log(`> Running socket on port: ${port}`);
    console.log('>> Waiting for connection...')
});

io.on('connection', (socket) => {
    socket.on('startTraining', async () => {
        console.log('>>> Loading models ...');
/*
        const startEpoch = 1;

        const generator     = md.createGenerator(ct.imageRows, ct.imagecols, ct.imageChannels, ct.inputDim);
        const discriminator = md.createDiscriminator(ct.imageRows, ct.imagecols, ct.imageChannels);
*/

        const startEpoch = tr.getPendingEpoch(ct.generatorHandlingDirectory);

        const generator     = await md.loadModel(ct.generatorHandlingDirectory, ct.generatorSavingDirectory);
        const discriminator = await md.loadModel(ct.discriminatorHandlingDirectory, ct.discriminatorSavingDirectory);

        const adversarial   = md.createAdversarial(generator, discriminator);

        md.compilModels(generator, discriminator, adversarial);

        const endEpoch = startEpoch + ct.nbEpochs;

        console.log('>>> Models loaded!');
        console.log('>>>> Training in progress ...');

        for(let epoch = startEpoch; epoch < endEpoch; epoch++) {
            console.log(`>>>> =========================== Epoch: ${epoch} ================================`);

            await tr.trainModels(generator, discriminator, adversarial, ct.nbEpochs, ct.nbImages, ct.trainingBatchSize, ct.inputDim);

            console.log(`>>>> ======================================================================`);
            console.log('');

            await md.saveModel(generator, ct.generatorHandlingDirectory, ct.generatorSavingDirectory, epoch);
            await md.saveModel(discriminator, ct.discriminatorHandlingDirectory, ct.discriminatorSavingDirectory, epoch);

            const batchImageArray = tr.generateImage(generator, ct.generatingBatchSize, ct.inputDim);
            io.emit('imageGenerated', batchImageArray);
        }

        console.log('>>>> Training completed!');
    });
});

