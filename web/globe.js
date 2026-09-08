// 3D WebGL Holographic Cyber Globe (Three.js Engine)
// Visualizes Cross-Border Hawala & International Money Mule Networks

let globeScene, globeCamera, globeRenderer;
let globeMesh, atmosphereMesh, arcGroup, cityGroup, particleGroup;
let isGlobeActive = false;
let isDraggingGlobe = false;
let previousMousePosition = { x: 0, y: 0 };
let globeContainer;

// Major Cross-Border Hawala & AML Hubs
const globalCities = [
    { name: 'Mumbai (Origin Phish)', lat: 19.0760, lon: 72.8777, type: 'victim', color: 0x38bdf8, vol: '?45 Lakhs' },
    { name: 'Delhi (Student Mule Hub)', lat: 28.6139, lon: 77.2090, type: 'mule', color: 0xef4444, vol: '?42 Lakhs' },
    { name: 'Dubai (Hawala / OTC Desk)', lat: 25.2048, lon: 55.2708, type: 'layer', color: 0xf59e0b, vol: 'AED 1.8M' },
    { name: 'Zurich (Private Layer)', lat: 47.3769, lon: 8.5417, type: 'layer', color: 0xa855f7, vol: 'CHF 410k' },
    { name: 'London (Shell Aggregate)', lat: 51.5074, lon: -0.1278, type: 'mule', color: 0xef4444, vol: 'GBP 350k' },
    { name: 'Singapore (Transit Rail)', lat: 1.3521, lon: 103.8198, type: 'layer', color: 0x38bdf8, vol: 'SGD 620k' },
    { name: 'Hong Kong (Crypto Cashout)', lat: 22.3193, lon: 114.1694, type: 'cashout', color: 0xef4444, vol: 'USDT 480k' },
    { name: 'New York (Clearing House)', lat: 40.7128, lon: -74.0060, type: 'bank', color: 0x10b981, vol: 'USD 500k' }
];

const crossBorderRoutes = [
    { from: 0, to: 1, type: 'illicit', label: 'Domestic Structuring' },
    { from: 1, to: 2, type: 'hawala', label: 'Cross-Border Hawala Hop' },
    { from: 2, to: 3, type: 'layer', label: 'Layering Wire Transfer' },
    { from: 3, to: 4, type: 'illicit', label: 'Shell Corp Placement' },
    { from: 1, to: 5, type: 'hawala', label: 'Southeast Asia Route' },
    { from: 5, to: 6, type: 'crypto', label: 'Crypto P2P Exit' }
];

function latLonToVector3(lat, lon, radius) {
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = (lon + 180) * (Math.PI / 180);
    const x = -(radius * Math.sin(phi) * Math.cos(theta));
    const z = (radius * Math.sin(phi) * Math.sin(theta));
    const y = (radius * Math.cos(phi));
    return new THREE.Vector3(x, y, z);
}

function initGlobe() {
    globeContainer = document.getElementById('globeContainer');
    if (!globeContainer || typeof THREE === 'undefined') return;

    const width = globeContainer.clientWidth || 600;
    const height = globeContainer.clientHeight || 500;

    globeScene = new THREE.Scene();
    globeCamera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    globeCamera.position.z = 240;

    globeRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    globeRenderer.setSize(width, height);
    globeRenderer.setPixelRatio(window.devicePixelRatio);
    globeContainer.innerHTML = '';
    globeContainer.appendChild(globeRenderer.domElement);

    // Starfield Background
    const starGeometry = new THREE.BufferGeometry();
    const starCount = 800;
    const starPositions = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount * 3; i++) {
        starPositions[i] = (Math.random() - 0.5) * 800;
    }
    starGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
    const starMaterial = new THREE.PointsMaterial({ color: 0x38bdf8, size: 1.2, transparent: true, opacity: 0.6 });
    const stars = new THREE.Points(starGeometry, starMaterial);
    globeScene.add(stars);

    // Wireframe Cyber Globe Sphere
    const globeRadius = 70;
    const sphereGeo = new THREE.SphereGeometry(globeRadius, 36, 36);
    const sphereMat = new THREE.MeshBasicMaterial({
        color: 0x0284c7,
        wireframe: true,
        transparent: true,
        opacity: 0.22
    });
    globeMesh = new THREE.Mesh(sphereGeo, sphereMat);
    globeScene.add(globeMesh);

    // Outer Atmospheric Glow
    const atmoGeo = new THREE.SphereGeometry(globeRadius * 1.05, 32, 32);
    const atmoMat = new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        wireframe: false,
        transparent: true,
        opacity: 0.05
    });
    atmosphereMesh = new THREE.Mesh(atmoGeo, atmoMat);
    globeScene.add(atmosphereMesh);

    // City Hub Markers
    cityGroup = new THREE.Group();
    globalCities.forEach((city) => {
        const pos = latLonToVector3(city.lat, city.lon, globeRadius);
        const pinGeo = new THREE.SphereGeometry(1.8, 12, 12);
        const pinMat = new THREE.MeshBasicMaterial({ color: city.color });
        const pin = new THREE.Mesh(pinGeo, pinMat);
        pin.position.copy(pos);
        cityGroup.add(pin);

        // Halo Ring
        const ringGeo = new THREE.RingGeometry(2.5, 3.5, 16);
        const ringMat = new THREE.MeshBasicMaterial({ color: city.color, side: THREE.DoubleSide, transparent: true, opacity: 0.6 });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.position.copy(pos);
        ring.lookAt(new THREE.Vector3(0, 0, 0));
        cityGroup.add(ring);
    });
    globeMesh.add(cityGroup);

    // 3D Ballistic Bezier Curves
    arcGroup = new THREE.Group();
    crossBorderRoutes.forEach(route => {
        const u = globalCities[route.from];
        const v = globalCities[route.to];
        const p1 = latLonToVector3(u.lat, u.lon, globeRadius);
        const p2 = latLonToVector3(v.lat, v.lon, globeRadius);

        // Control point arching into space
        const mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
        const dist = p1.distanceTo(p2);
        mid.normalize().multiplyScalar(globeRadius + dist * 0.35);

        const curve = new THREE.QuadraticBezierCurve3(p1, mid, p2);
        const points = curve.getPoints(40);
        const arcGeo = new THREE.BufferGeometry().setFromPoints(points);
        const arcMat = new THREE.LineBasicMaterial({
            color: route.type === 'hawala' ? 0xf59e0b : 0xef4444,
            transparent: true,
            opacity: 0.75,
            linewidth: 2
        });
        const arcLine = new THREE.Line(arcGeo, arcMat);
        arcGroup.add(arcLine);
    });
    globeMesh.add(arcGroup);

    // Mouse Interaction for 3D Rotation
    globeContainer.addEventListener('mousedown', (e) => {
        isDraggingGlobe = true;
        previousMousePosition = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener('mousemove', (e) => {
        if (!isDraggingGlobe || !globeMesh) return;
        const deltaX = e.clientX - previousMousePosition.x;
        const deltaY = e.clientY - previousMousePosition.y;

        globeMesh.rotation.y += deltaX * 0.006;
        globeMesh.rotation.x += deltaY * 0.006;

        previousMousePosition = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener('mouseup', () => {
        isDraggingGlobe = false;
    });

    globeContainer.addEventListener('wheel', (e) => {
        globeCamera.position.z += e.deltaY * 0.15;
        globeCamera.position.z = Math.max(140, Math.min(360, globeCamera.position.z));
        e.preventDefault();
    }, { passive: false });

    animateGlobe();
}

function animateGlobe() {
    if (!isGlobeActive) return;
    requestAnimationFrame(animateGlobe);

    if (globeMesh && !isDraggingGlobe) {
        globeMesh.rotation.y += 0.002;
    }

    globeRenderer.render(globeScene, globeCamera);
}

function toggleGlobeView() {
    const canvas = document.getElementById('networkCanvas');
    const globeCont = document.getElementById('globeContainer');
    const viewBtn = document.getElementById('globeToggleBtn');
    audio.playClick();

    isGlobeActive = !isGlobeActive;
    if (isGlobeActive) {
        canvas.style.display = 'none';
        globeCont.style.display = 'block';
        viewBtn.innerText = '??? 2D Force Graph';
        document.getElementById('canvasStatusText').innerText = '3D WebGL Cross-Border Hawala Tracker Active';
        if (!globeRenderer) {
            initGlobe();
        } else {
            animateGlobe();
        }
        addChatMessage('?? <b>3D WebGL Cyber Globe Initialized:</b> Visualizing cross-border Hawala structuring pipelines spanning Mumbai, Dubai, Zurich, and Hong Kong.', 'bot');
    } else {
        canvas.style.display = 'block';
        globeCont.style.display = 'none';
        viewBtn.innerText = '?? 3D Cyber Globe';
        document.getElementById('canvasStatusText').innerText = 'Graph Physics Engine Active (60 FPS)';
        addChatMessage('??? Switched back to 2D Force-Directed Node Graph.', 'bot');
    }
}

