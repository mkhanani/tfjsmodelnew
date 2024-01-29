const tf = require('@tensorflow/tfjs');

class AI {

    // Compile Model
    compile() {

        const model = tf.sequential();

        // Input layer
        model.add(tf.layers.dense({
            units: 3,
            inputShape: [3]
        }))

        //Output Layer
        model.add(tf.layers.dense({
            units: 2
        }))

        model.compile({
            loss: 'meanSquaredError',
            optimizer: 'sgd'
        })

        return model;
    }

    //Run Model/predict
    run() {

        const model = this.compile();

        // xs - input later
        const xs = tf.tensor2d([
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3]
        ]);

        // ys - Output layer
        const ys = tf.tensor2d([
            [1, 0],
            [0, 1],
            [1, 1]
        ]);

        model.fit(xs, ys, {
            epochs: 1000

        }).then(() => {

            const data = tf.tensor2d([
                [1.0, 1.0, 1.0]
            ])

            const prediction = model.predict(data);
            prediction.print();
        });


    }
}

const ai = new AI();
ai.run();