const { loadImage, createCanvas } = require('canvas')
const myimg = loadImage("./36.jpg")
const  tf = require ('@tensorflow/tfjs');
global.fetch = require('node-fetch');
const IMAGE_SIZE = 224;

const mobileNet = loadMobilenet();

async function loadMobilenet() {
    const model = await tf.loadModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  
    // Return a model that outputs an internal activation.
    const layer = model.getLayer('conv_pw_13_relu');
    return tf.model({inputs: model.inputs, outputs: layer.output});
}

async function imageLoader(path) {
    return new Promise((resolve, reject) => {
        loadImage(path)
        .then(img => {
            const canvas = createCanvas(img.width, img.height)
            const ctx = canvas.getContext('2d')
            ctx.drawImage(img, 0,0)
            resolve(createBatch(canvas))
        })
        .catch(err => reject(err))
    })
}


async function createBatch(canvas) {
    
    const raw = tf.fromPixels(canvas).toFloat();
    const cropped = cropImage(raw); 
    const resized = tf.image.resizeBilinear(cropped, [IMAGE_SIZE, IMAGE_SIZE])
    
    // Normalize the image from [0, 255] to [-1, 1].
    const offset = tf.scalar(127);
    const normalized = resized.sub(offset).div(offset);
    
    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.expandDims(0);
    return batched;
}


function cropImage(img) {
  const size = Math.min(img.shape[0], img.shape[1]);
  const centerHeight = img.shape[0] / 2;
  const beginHeight = centerHeight - (size / 2);
  const centerWidth = img.shape[1] / 2;
  const beginWidth = centerWidth - (size / 2);
  return img.slice([Math.round(beginHeight), Math.round(beginWidth), 0], [size, size, 3]);
}

module.exports = {
    imageLoader:imageLoader,
    loadMobilenet: loadMobilenet
}