// Initialize map
const initialCoords = [37.5, -4.5];
const initialZoom = 7;
const map = L.map('map').setView(initialCoords, initialZoom);

// Add OpenStreetMap tiles
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors'
}).addTo(map);

// Marker for selected location (none initially)
let marker = null;

const latInput = document.getElementById('latitude');
const lngInput = document.getElementById('longitude');
const form = document.getElementById('prediction-form');
const errorAlert = document.getElementById('error-alert');
const resultCard = document.getElementById('result-card');
const predictedPriceSpan = document.getElementById('predicted-price');

// Handle map clicks to select location
map.on('click', (e) => {
  const lat = e.latlng.lat;
  const lng = e.latlng.lng;
  // If a marker already exists, move it; otherwise, create a new one
  if (marker) {
    marker.setLatLng([lat, lng]);
  } else {
    marker = L.marker([lat, lng]).addTo(map);
  }
  // Update latitude and longitude input fields (6 decimal places)
  latInput.value = lat.toFixed(6);
  lngInput.value = lng.toFixed(6);
  // Clear any custom validation error for lat/long now that both are provided
  latInput.setCustomValidity('');
  lngInput.setCustomValidity('');
});

// Form submission handling
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  // Remove previous alerts or results
  errorAlert.classList.add('d-none');
  errorAlert.innerText = '';
  // Hide result card if visible
  resultCard.classList.add('d-none');
  resultCard.classList.remove('result-visible');

  // Custom validation: if one of lat or lng is filled, both must be filled
  if ((latInput.value && !lngInput.value) || (!latInput.value && lngInput.value)) {
    latInput.setCustomValidity('Latitud y longitud deben especificarse juntas.');
    lngInput.setCustomValidity('Latitud y longitud deben especificarse juntas.');
  } else {
    latInput.setCustomValidity('');
    lngInput.setCustomValidity('');
  }

  // Check overall form validity
  if (!form.checkValidity()) {
    event.stopPropagation();
    // Show validation feedback
    form.classList.add('was-validated');
    return;
  }

  // Gather form data
  const surface = parseFloat(document.getElementById('surface').value);
  const rooms = parseInt(document.getElementById('rooms').value);
  const bathrooms = parseInt(document.getElementById('bathrooms').value);
  const propertyType = document.getElementById('propertyType').value;
  const province = document.getElementById('province').value;
  // Handle optional lat/long
  let latVal = latInput.value.trim();
  let lngVal = lngInput.value.trim();
  latVal = latVal === '' ? null : parseFloat(latVal);
  lngVal = lngVal === '' ? null : parseFloat(lngVal);

  const requestData = {
    'superficie': surface,
    'habitaciones': rooms,
    'baños': bathrooms,
    'tipo_propiedad': propertyType,
    'latitud': latVal,
    'longitud': lngVal,
    'provincia': province
  };

  try {
    const response = await fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestData)
    });
    if (!response.ok) {
      // Try to extract error message from response
      let errorMsg = 'Error al obtener la predicción.';
      try {
        const errorData = await response.json();
        if (errorData.error) {
          errorMsg = errorData.error;
        } else if (response.status === 500) {
          errorMsg = 'Error del servidor. Inténtalo más tarde.';
        }
      } catch (e) {
        if (response.status === 404) {
          errorMsg = 'No se encontró la API de predicción.';
        }
      }
      // Show error alert
      errorAlert.innerText = errorMsg;
      errorAlert.classList.remove('d-none');
      return;
    }
    const data = await response.json();
    if (data.prediction === undefined) {
      throw new Error('Respuesta inesperada del servidor.');
    }
    // Format prediction result
    const predictedValue = data.prediction;
    const roundedValue = Math.round(predictedValue);
    const formattedPrice = new Intl.NumberFormat('es-ES', {
      style: 'currency',
      currency: 'EUR',
      maximumFractionDigits: 0
    }).format(roundedValue);
    // Display result
    predictedPriceSpan.innerText = formattedPrice;
    // Show result card with animation
    resultCard.classList.remove('d-none');
    // Trigger reflow for transition
    void resultCard.offsetWidth;
    resultCard.classList.add('result-visible');
  } catch (error) {
    // Network or parsing error
    console.error('Prediction error:', error);
    errorAlert.innerText = 'No se pudo conectar con la API de predicción.';
    errorAlert.classList.remove('d-none');
  }
});

// Clear button handling
document.getElementById('clear-btn').addEventListener('click', () => {
  form.reset();
  form.classList.remove('was-validated');
  // Hide and reset result and error message
  errorAlert.classList.add('d-none');
  errorAlert.innerText = '';
  resultCard.classList.add('d-none');
  resultCard.classList.remove('result-visible');
  predictedPriceSpan.innerText = '';
  // Remove map marker and reset view
  if (marker) {
    map.removeLayer(marker);
    marker = null;
  }
  map.setView(initialCoords, initialZoom);
});
