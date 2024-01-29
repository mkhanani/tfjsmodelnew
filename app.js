document.addEventListener('DOMContentLoaded', () => {
    const imageInput = document.getElementById('image-input');
    const predictButton = document.getElementById('predict-button');
    const resultDiv = document.getElementById('result');

    // Load the model
    let model;
    (async () => {
        model = await tf.loadLayersModel('https://mkhanani.github.io/tfjsmodel/model.json');
    })();

    // Predict function
    async function predict(input) {
        const resizedImage = tf.image.resizeBilinear(input, [28, 28]);
        const normalizedImage = resizedImage.div(tf.scalar(255));
        const batchedImage = normalizedImage.expandDims(0);
        const prediction = model.predict(batchedImage);
        const topClass = prediction.argMax(-1);
        const classNames = ['class1', 'class2', 'class3']; // Replace with your class names
        const classIndex = await topClass.data();
        resultDiv.textContent = `Predicted class: ${classNames[classIndex]}`;
    }

    // File input change event
    imageInput.addEventListener('change', () => {
        const file = imageInput.files[0];
        const reader = new FileReader();

        reader.onload = (event) => {
            const img = new Image();
            img.src = event.target.result;
            img.onload = () => {
                predict(tf.browser.fromPixels(img));
            };
        };

        reader.readAsDataURL(file);
    });

    // Predict button click event
    predictButton.addEventListener('click', () => {
        if (imageInput.files.length > 0) {
            predict(tf.browser.fromPixels(imageInput.files[0]));
        } else {
            resultDiv.textContent = 'Please select an image first.';
        }
    });
});