// Este script se cargará después de que el DOM principal y los parciales estén listos (controlado desde predictor.html)

const form = document.getElementById('predictionForm'); // ID del script original
const resultDiv = document.getElementById('result'); // ID del script original para mostrar resultados
const latitudInput = document.getElementById('latitud'); // ID del script original
const longitudInput = document.getElementById('longitud'); // ID del script original
const mapElement = document.getElementById('map');

// Elementos para mensajes de error de Tailwind (si los mantenemos)
const errorAlertGeneral = document.getElementById('error-alert'); // Para errores generales de API
const latLngErrorDivSpecific = document.getElementById('latlng-error'); // Para error específico de lat/lng

if (!mapElement || !form) {
    console.error("Faltan elementos esenciales del DOM (mapa o formulario) para inicializar app.js");
} else {
    // 1) Definimos los límites aproximados de Andalucía
    const andaluciaBounds = [
        [36.00, -7.80],   // esquina sudoeste (lat, lng)
        [38.80, -0.10]    // esquina noreste  (lat, lng)
    ];

    // 2) Inicializamos el mapa “encerrado” solo en Andalucía
    const map = L.map('map', {
        // maxBounds: andaluciaBounds, // Descomentar si se quiere restringir estrictamente
        // maxBoundsViscosity: 1.0, 
        minZoom: 7
    });

    // 3) Cargamos la capa base de OpenStreetMap
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    // 4) Ajustamos la vista inicial para encuadrar toda Andalucía o usar coords iniciales
    const initialLatValue = parseFloat(latitudInput.value);
    const initialLngValue = parseFloat(longitudInput.value);

    if (!isNaN(initialLatValue) && !isNaN(initialLngValue)) {
        map.setView([initialLatValue, initialLngValue], 13); // Zoom más cercano si hay coords
    } else {
        map.fitBounds(andaluciaBounds);
    }

    let marker;

    // Función para actualizar los campos del formulario y el marcador
    function updateCoordinates(lat, lng, setView = true) {
        latitudInput.value = lat.toFixed(10); // Precisión del script original
        longitudInput.value = lng.toFixed(10); // Precisión del script original

        if (marker) {
            map.removeLayer(marker);
        }
        marker = L.marker([lat, lng]).addTo(map);
        if (setView) {
            map.setView([lat, lng], map.getZoom() < 13 ? 13 : map.getZoom()); // Ajustar zoom si es necesario
        }
        // Ocultar error de lat/lng si se seleccionan en el mapa
        if (latLngErrorDivSpecific) latLngErrorDivSpecific.classList.add('hidden');
        // Disparar evento input para validación en tiempo real si existe
        latitudInput.dispatchEvent(new Event('input'));
        longitudInput.dispatchEvent(new Event('input'));
    }

    // Si los inputs ya traían coordenadas al cargar la página, colocamos el marcador
    if (!isNaN(initialLatValue) && !isNaN(initialLngValue)) {
        updateCoordinates(initialLatValue, initialLngValue, false); // false para no cambiar la vista inicial ya establecida
    }

    // Evento de clic en el mapa
    map.on('click', function(e) {
        updateCoordinates(e.latlng.lat, e.latlng.lng);
    });

    // Funciones de validación de Tailwind (opcional mantenerlas si se quieren mensajes más detallados)
    function validateField(field) {
        const errorDiv = field.parentElement.querySelector('.invalid-feedback');
        if (!errorDiv) return;
        if (field.checkValidity()) {
            errorDiv.classList.add('hidden');
            field.classList.remove('border-red-500'); field.classList.add('border-gray-300');
        } else {
            errorDiv.classList.remove('hidden');
            field.classList.add('border-red-500'); field.classList.remove('border-gray-300');
        }
    }

    function validateLatLngInputs() {
        const latVal = latitudInput.value.trim();
        const lngVal = longitudInput.value.trim();
        let isValid = true;
        if ((latVal && !lngVal) || (!latVal && lngVal)) {
            if (latLngErrorDivSpecific) latLngErrorDivSpecific.classList.remove('hidden');
            latitudInput.classList.add('border-red-500'); longitudInput.classList.add('border-red-500');
            isValid = false;
        } else {
            if (latLngErrorDivSpecific) latLngErrorDivSpecific.classList.add('hidden');
            if (latVal && lngVal) { // Solo quitar borde si ambos están llenos o ambos vacíos
                latitudInput.classList.remove('border-red-500'); longitudInput.classList.remove('border-red-500');
                latitudInput.classList.add('border-gray-300'); longitudInput.classList.add('border-gray-300');
            }
        }
        return isValid;
    }
    
    // Añadir listeners para validación en tiempo real (opcional)
    Array.from(form.elements).forEach(element => {
        if (element.required || element.id === 'latitud' || element.id === 'longitud') { 
            element.addEventListener('input', () => {
                if (element.required) validateField(element); // Valida campos requeridos
                if (element.id === 'latitud' || element.id === 'longitud') validateLatLngInputs(); // Valida la pareja lat/lng
            });
        }
    });


    form.addEventListener('submit', async function(event) {
        event.preventDefault(); 
        
        // Limpiar mensajes previos
        if (resultDiv) resultDiv.innerHTML = 'Calculando predicción...';
        if (errorAlertGeneral) errorAlertGeneral.classList.add('hidden');

        // Validación básica (HTML5 checkValidity y la de lat/lng)
        let formIsValid = form.checkValidity();
        if (!validateLatLngInputs()) { // Asegura que lat/lng son válidos como pareja
            formIsValid = false;
        }
        
        // Aplicar estilos de validación de Tailwind si se desea
        Array.from(form.elements).forEach(el => { if(el.required) validateField(el); });


        if (!formIsValid) {
            if (resultDiv) resultDiv.innerHTML = '<p class="text-red-500">Por favor, corrige los errores en el formulario.</p>';
            return;
        }

        const formData = new FormData(form);
        const data = {};

        formData.forEach((value, key) => {
            // El script original convierte estos a float, el backend espera números
            if (['superficie', 'habitaciones', 'baños', 'latitud', 'longitud'].includes(key)) {
                data[key] = parseFloat(value);
            } else {
                data[key] = value;
            }
        });
        
        // Añadir provincia si existe en el formulario (el script original no la tenía, pero nuestro form sí)
        if (document.getElementById('province')) {
            data['provincia'] = document.getElementById('province').value;
        }


        try {
            // Usar la URL del backend correcta (relativa o absoluta)
            const response = await fetch('/predict', { // Ajustado a ruta relativa
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (response.ok) {
                if (result.prediction !== undefined) {
                    if (resultDiv) {
                        resultDiv.innerHTML = `
                            <h3 class="text-xl font-semibold text-green-700 mb-2">Precio Estimado:</h3>
                            <p class="text-3xl font-bold text-green-600">€ ${parseFloat(result.prediction).toLocaleString('es-ES', {
                                minimumFractionDigits: 0, // Sin decimales para el precio
                                maximumFractionDigits: 0
                            })}</p>
                        `;
                        // Hacer visible el card de resultado si estaba oculto por Tailwind
                        resultDiv.classList.remove('hidden'); 
                        resultDiv.classList.add('result-visible'); // Para la animación si se usa
                    }
                } else {
                    if (resultDiv) resultDiv.innerHTML = `<p class="text-red-600">Respuesta inesperada del servidor.</p>`;
                }
            } else {
                if (resultDiv) resultDiv.innerHTML = `<p class="text-red-600">Error: ${result.error || response.statusText}</p>`;
            }

        } catch (error) {
            console.error('Error al realizar la petición:', error);
            if (resultDiv) resultDiv.innerHTML = `<p class="text-red-600">Error de conexión o al procesar la solicitud. Revisa la consola.</p>`;
        }
    });

    const clearBtn = document.getElementById('clear-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            form.reset(); // Resetea a los values definidos en el HTML
            if (resultDiv) resultDiv.innerHTML = ''; // Limpiar resultado
            if (resultDiv) resultDiv.classList.add('hidden'); // Ocultar card de resultado
            if (errorAlertGeneral) errorAlertGeneral.classList.add('hidden'); // Ocultar alerta general
            if (latLngErrorDivSpecific) latLngErrorDivSpecific.classList.add('hidden'); // Ocultar error lat/lng

            // Limpiar estilos de validación de Tailwind
            Array.from(form.elements).forEach(element => {
                const errorDivFeedback = element.parentElement.querySelector('.invalid-feedback');
                if (errorDivFeedback) errorDivFeedback.classList.add('hidden');
                element.classList.remove('border-red-500');
                element.classList.add('border-gray-300');
            });

            // Resetear el mapa
            const defaultLat = parseFloat(latitudInput.defaultValue) || initialLatValue; // Usa defaultValue para el reset
            const defaultLng = parseFloat(longitudInput.defaultValue) || initialLngValue; // Usa defaultValue para el reset
            
            if (marker) {
                map.removeLayer(marker);
                marker = null;
            }
            if (!isNaN(defaultLat) && !isNaN(defaultLng)) {
                updateCoordinates(defaultLat, defaultLng);
                map.setView([defaultLat, defaultLng], 13);
            } else {
                map.fitBounds(andaluciaBounds);
            }
        });
    }
}
