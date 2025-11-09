document.addEventListener('DOMContentLoaded', function() {
    const neurons = document.querySelectorAll('.neuron');
    const connections = document.querySelectorAll('.connection');
    
    // Detect if device supports touch
    const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    
    // --- 1. Animate neurons fading in ---
    neurons.forEach((neuron, index) => {
        neuron.style.opacity = '0';
        setTimeout(() => {
            neuron.style.transition = 'opacity 0.5s ease';
            neuron.style.opacity = '1';
        }, index * 100);
    });

    // --- 2. Smart connection highlighting ---
    function highlightConnections(neuronId) {
        // Fade out ALL connections first
        connections.forEach(conn => {
            conn.style.opacity = '0.1';
            conn.classList.remove('highlight');
        });
        
        // Select and highlight only the connected lines
        const relevantConnections = document.querySelectorAll(`.${neuronId}`);
        relevantConnections.forEach(conn => {
            conn.style.opacity = '1';
            conn.classList.add('highlight');
        });
    }
    
    function resetConnections() {
        connections.forEach(conn => {
            conn.style.opacity = '0.6';
            conn.classList.remove('highlight');
        });
    }
    
    neurons.forEach(neuron => {
        if (isTouchDevice) {
            // Touch events for mobile
            neuron.addEventListener('touchstart', (e) => {
                e.preventDefault();
                const neuronId = neuron.id;
                highlightConnections(neuronId);
            });
            
            neuron.addEventListener('touchend', (e) => {
                e.preventDefault();
                setTimeout(resetConnections, 1000); // Keep highlighted for 1 second
            });
        } else {
            // Mouse events for desktop
            neuron.addEventListener('mouseenter', () => {
                const neuronId = neuron.id;
                highlightConnections(neuronId);
            });
            
            neuron.addEventListener('mouseleave', () => {
                resetConnections();
            });
        }
    });
    
    // Reset on touch outside neurons
    if (isTouchDevice) {
        document.addEventListener('touchstart', (e) => {
            if (!e.target.classList.contains('neuron')) {
                resetConnections();
            }
        });
    }
});
```

## File Structure

Create these 3 files in the same folder:
```
project-folder/
├── index.html
├── styles.css
└── script.js
