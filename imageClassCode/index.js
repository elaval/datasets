const { loadImage, createCanvas } = require('canvas')
const myimg = loadImage("./36.jpg")
const  tf = require ('@tensorflow/tfjs');
//require('@tensorflow/tfjs-node');
global.fetch = require('node-fetch');

const dataSet = require("./dataSet");





myimg.then((img) => {
  const canvas = createCanvas(img.width, img.height)
  const ctx = canvas.getContext('2d')
  ctx.drawImage(img, 0,0)
  processImage(canvas);
}).catch(err => {
  console.log('oh no!', err)
})

async function processImage(img) {
  const IMAGE_SIZE = 224;
  const raw = tf.fromPixels(img).toFloat();
  const cropped = cropImage(raw); 
  const resized = tf.image.resizeBilinear(cropped, [IMAGE_SIZE, IMAGE_SIZE])
  
  // Normalize the image from [0, 255] to [-1, 1].
  const offset = tf.scalar(127);
  const normalized = resized.sub(offset).div(offset);
  
  // Reshape to a single-element batch so we can pass it to predict.
  const batched = normalized.expandDims(0);
  console.log(batched)

  const premodel = await loadMobilenet();
  const newInput = premodel.predict(batched);

  const model = newModel();

  const batchSize = 5

  model.fit([newInput], tf.oneHot([1],2), {
    batchSize,
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        // Log the cost for every batch that is fed.
        console.log('Cost: ' + logs.loss.toFixed(5));
        //await tf.nextFrame();
      }
    }
  })
  .then((d) => {
    console.log(model.predict([newInput]).dataSync());
    console.log(d);
  })
  
}

function newModel() {
  trainableModel = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: [7, 7, 256]}),
      tf.layers.dense({
        units: 8,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      tf.layers.dense({
        units: 2,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  })

  trainableModel.compile({optimizer: 'adam', loss: 'categoricalCrossentropy'});

  return trainableModel;
}

function cropImage(img) {
  const size = Math.min(img.shape[0], img.shape[1]);
  const centerHeight = img.shape[0] / 2;
  const beginHeight = centerHeight - (size / 2);
  const centerWidth = img.shape[1] / 2;
  const beginWidth = centerWidth - (size / 2);
  return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
}

async function loadMobilenet() {
  const model = await tf.loadModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Return a model that outputs an internal activation.
  const layer = model.getLayer('conv_pw_13_relu');
  return tf.model({inputs: model.inputs, outputs: layer.output});
}