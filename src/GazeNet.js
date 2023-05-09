import * as tf from "@tensorflow/tfjs";
import { StandardScaler } from "./Scaler";

export class GazeNet {
    constructor() {
        this.trained = false;
        this.scalerX = new StandardScaler(45);
        this.scalerY = new StandardScaler(45);
        this.train_data = [];
        this.labels = [];
        this.model = tf.sequential({
            layers: [
                tf.layers.dense({ inputShape: (124), units: 128, activation: 'relu' }),
                tf.layers.dense({ units: 64, activation: 'relu' }),
                tf.layers.dense({ units: 32, activation: 'relu' }),
                tf.layers.dense({ units: 2, activation: 'linear' })
            ]
        });
        const myOptimizer = tf.train.adam(.001)
        this.model.compile({ loss: 'meanSquaredError', optimizer: myOptimizer });
    }

    async train(data, labels) {
        if (this.train_data.length >= 40) {
            this.scalerX = new StandardScaler(0)
            this.scalerY = new StandardScaler(0)
            const x = tf.tensor2d(this.train_data);
            const y = tf.tensor2d(this.labels);
            const x_scaled = this.scalerX.fitTransform(x)
            const y_scaled = this.scalerY.fitTransform(y)
            this.model.fit(x_scaled, y_scaled, {
                epochs: 10,
                batchSize: 5,
                metrics: ['accuracy']
            })
            this.trained = true;
            this.train_data = []
            this.labels = []
        }
        else {
            const entry_data = this.preprocess(data);

            for (let i = 0; i < 5; i++) {
                this.train_data.push(entry_data)
                this.labels.push(labels)
            }

        }


    }
    predict(data) {
        if (!this.trained) {
            return [window.innerWidth / 2, window.innerHeight / 2];
        }
        const data_processed = this.preprocess(data)
        const x = tf.tensor(data_processed);
        const x_scaled = this.scalerX.transform(x);

        const x_reshaped = x_scaled.reshape([1, data_processed.length]);
        const prediction_scaled = this.model.predict(x_reshaped);
        const prediction = this.scalerY.inverseTransform(prediction_scaled);
        return prediction.dataSync();
    }

    preprocess = (face) => {
        const annots = face.annotations
        const points3d = [...annots.leftEyeIris, ...annots.leftEyeLower0, ...annots.leftEyeLower1, ...annots.leftEyebrowUpper,
        ...annots.rightEyeIris, ...annots.rightEyeLower0, ...annots.rightEyeLower1, ...annots.rightEyebrowUpper]
        const points2d = points3d.map(arr => arr.slice(0, 2));
        return points2d.flat();
    }

}