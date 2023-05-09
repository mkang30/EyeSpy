import * as tf from "@tensorflow/tfjs";

export class StandardScaler {
    constructor(size) {
        this.mean = null;
        this.variance = null;
        this.size = size;
    }

    fit(X) {
        const { mean, variance } = tf.moments(X, 0);
        if (this.mean == null) {
            this.mean = mean;
            this.variance = variance;
        }
        else {
            this.mean = (this.mean + mean) / 2.0
            this.variance = ((this.size - 1) * this.variance + (this.size - 1) * variance) / (this.size + this.size - 2)
        }
    }

    transform(X) {
        const scaled = tf.sub(X, this.mean).div(tf.sqrt(this.variance));
        return scaled;
    }

    fitTransform(X) {
        this.fit(X);
        return this.transform(X);
    }

    inverseTransform(X) {
        const unscaled = tf.mul(tf.sqrt(this.variance), X).add(this.mean);
        return unscaled;
    }
}
