document.getElementById("submitBtn").addEventListener("click", function() {
    const imageInput = document.getElementById("imageUpload");
    const messageElement = document.getElementById("message");
    const resultElement = document.getElementById("result");

    // Clear previous messages
    messageElement.textContent = "";
    resultElement.textContent = "";

    if (!imageInput.files.length) {
        // Display warning if no file is selected
        messageElement.textContent = "Please upload an image";
    } else {
        // Show loading message while processing
        resultElement.textContent = "Processing...";

        // Simulate sending the image for detection (replace with real API call)
        setTimeout(() => {
            // Simulate the response (replace this logic with your backend API call)
            const isPneumonia = Math.random() > 0.5; // Randomly returns positive or negative

            // Display the result
            if (isPneumonia) {
                resultElement.textContent = "Pneumonia Positive";
                resultElement.style.color = "red";
            } else {
                resultElement.textContent = "Pneumonia Negative";
                resultElement.style.color = "green";
            }
        }, 2000); // Simulated delay
    }
});