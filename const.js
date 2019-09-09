const imageRows     = 80;
const imagecols     = 120;
const imageChannels = 1;
const inputDim      = 100;

const trainingBatchSize   = 64;
const generatingBatchSize = 10
const nbImages            = 12186;

const nbEpochs = 1000;

const savingPath = 'file:///Developpement/Node/BigTellOfTheconquest/Models';
const handlingPath = 'D:/Developpement/Node/BigTellOfTheconquest/Models';

const discriminatorSavingDirectory = `${savingPath}/Discriminator`;
const generatorSavingDirectory     = `${savingPath}/Generator`;

const discriminatorHandlingDirectory = `${handlingPath}/Discriminator`;
const generatorHandlingDirectory     = `${handlingPath}/Generator`;

module.exports = {
    imageRows,
    imagecols,
    imageChannels,
    inputDim,
    trainingBatchSize,
    generatingBatchSize,
    nbImages,
    nbEpochs,
    discriminatorSavingDirectory,
    generatorSavingDirectory,
    discriminatorHandlingDirectory,
    generatorHandlingDirectory
}