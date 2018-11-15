const dir = require('node-dir');
const { readdirSync, statSync } = require('fs');
const { join } = require('path');
const mn = require("./mobileNet");
const  tf = require ('@tensorflow/tfjs');

const dirs = p => readdirSync(p).filter(f => statSync(join(p, f)).isDirectory())
const files = p => readdirSync(p).filter(f => !statSync(join(p, f)).isDirectory())

const IMAGE_TYPES = ["K_II", "K_III_S", "K_III_L", "K_IV","K_V"];
//const IMAGE_TYPES = ["K_II"];

let newInput;


const BASE_PATH = "./data"

function getTypes() {
    return dirs(BASE_PATH)
}

function getImages(type) {
    return files(join(BASE_PATH, type))
}

async function setup() {
    const imageSet = [];
    const labels = [];
    const model = await mn.loadMobilenet();
    IMAGE_TYPES.forEach((type,i) => {
        console.log(type, i)
        const images = getImages(type);
        images.forEach(image => {
            imageSet.push(join(BASE_PATH, type, image));
            labels.push(i);
        })
        

    })


    const y=tf.oneHot(labels,IMAGE_TYPES.length)

    const xs = await mobileNetPartialRepresenations(model, imageSet)

    let x  = null

    xs.forEach(d => {
        if (x == null) {
            x = d;
        } else {
            x = x.concat(d, 0)
        }
    })

    const mymodel = newModel();

    const batchSize = 5

    mymodel.fit(x, y, {
        //batchSize,
        epochs: 5,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                // Log the cost for every batch that is fed.
               //console.log('Cost: ' + logs.loss.toFixed(5));
                //await tf.nextFrame();
            },
            onEpochEnd: async (num, logs) => {
                // Log the cost for every batch that is fed.
                console.log(num, 'Cost: ' + logs.loss.toFixed(5));
                //await tf.nextFrame();
            }
        }
    })
    .then((d) => {
        const p = mymodel.predict(x);
        console.log(p.argMax(1).dataSync());        
        console.log(d);
    })

}

async function mobileNetPartialRepresenations(model,images) {
    return new Promise((resolve, reject) => {
        const tensorPromises = images.map(image => process(image))
        Promise.all(tensorPromises)
        .then(imageTensors =>{
            const predictions = imageTensors.map(d => model.predict(d));
            resolve(predictions);
        })
        .catch(err => reject(err));
    })
}


setup();

function process(image) {
    return mn.imageLoader(image);
}


function newModel() {
    trainableModel = tf.sequential({
    layers: [
        tf.layers.flatten({inputShape: [7, 7, 256]}),
        tf.layers.dense({
        units: 100,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        //useBias: true
        }),
        tf.layers.dense({
        units: IMAGE_TYPES.length,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
        })
    ]
    })

    trainableModel.compile({optimizer: 'adam', loss: 'categoricalCrossentropy'});

    return trainableModel;
}


 
module.exports = {
    //getClasses : getClasses,
    getImages : getImages
}