document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const uploadButton = document.getElementById('uploadButton');
    const imagePreview = document.getElementById('imagePreview');
    const imagePreviewImg = imagePreview.querySelector('.image-preview__image');
    const imagePreviewText = imagePreview.querySelector('.image-preview__default-text');
    const loader = document.getElementById('loader');
    const errorDiv = document.getElementById('error');
    const resultsSection = document.getElementById('results');
    const resultsGrid = document.getElementById('resultsGrid');

    uploadButton.addEventListener('click', () => {
        imageUpload.click();
    });

    imageUpload.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (file) {
            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreviewImg.src = e.target.result;
                imagePreviewImg.style.display = 'block';
                imagePreviewText.style.display = 'none';
            };
            reader.readAsDataURL(file);

            // Reset UI
            errorDiv.style.display = 'none';
            resultsSection.style.display = 'none';
            resultsGrid.innerHTML = '';
            loader.style.display = 'block';

            // Upload and get recommendations
            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/recommend/', {
                    method: 'POST',
                    body: formData,
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'An unknown error occurred.');
                }

                const data = await response.json();
                displayResults(data.recommendations);

            } catch (error) {
                showError(error.message);
            } finally {
                loader.style.display = 'none';
            }
        }
    });

    function displayResults(recommendations) {
        if (recommendations && recommendations.length > 0) {
            resultsGrid.innerHTML = '';
            recommendations.forEach(item => {
                const resultItem = document.createElement('div');
                resultItem.className = 'result-item';
                
                const img = document.createElement('img');
                img.src = `/images/${item.filename}`;
                img.alt = item.filename;

                const p = document.createElement('p');
                p.textContent = `Dist: ${item.distance.toFixed(4)}`;

                resultItem.appendChild(img);
                resultItem.appendChild(p);
                resultsGrid.appendChild(resultItem);
            });
            resultsSection.style.display = 'block';
        } else {
            showError('No recommendations found.');
        }
    }

    function showError(message) {
        errorDiv.textContent = `Error: ${message}`;
        errorDiv.style.display = 'block';
        resultsSection.style.display = 'none';
    }
});
