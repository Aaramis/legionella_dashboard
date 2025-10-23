// VERSION DEBUG - À copier dans scripts.js après vérification

// Attribution data for proteins (will be injected by Python)
let attributionData = {};

// t-SNE plot references
let tsne2DDiv = null;
let tsne3DDiv = null;
let originalData2D = null;
let originalData3D = null;

// Initialize attribution data
function initAttributionData(data) {
    attributionData = data;
}

function addClickListener(plotDiv) {
    plotDiv.on('plotly_click', function(data) {
        if (data.points && data.points.length > 0) {
            const point = data.points[0];
            let proteinId = null;

            // Get protein ID from customdata
            if (point.customdata && point.customdata.length > 0) {
                proteinId = point.customdata[0];
            }

            console.log('Clicked protein ID:', proteinId);
            console.log('Available attributions:', Object.keys(attributionData));

            if (proteinId && attributionData[proteinId]) {
                displayAttribution(proteinId);
            } else if (proteinId) {
                displayNoAttribution(proteinId);
            }
        }
    });
}

function displayAttribution(proteinId) {
    const container = document.getElementById('attribution-container');
    if (!container) return;

    const figData = JSON.parse(attributionData[proteinId]);

    container.innerHTML = '';
    const plotDiv = document.createElement('div');
    plotDiv.id = 'attribution-plot';
    container.appendChild(plotDiv);

    Plotly.newPlot('attribution-plot', figData.data, figData.layout);

    const selector = document.getElementById('protein-selector');
    if (selector) {
        selector.value = proteinId;
    }

    container.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function displayNoAttribution(proteinId) {
    const container = document.getElementById('attribution-container');
    if (!container) return;

    container.innerHTML = `
        <div style="text-align: center; padding: 40px;">
            <h3 style="color: #666;">No attribution data available for: ${proteinId || 'this protein'}</h3>
            <p style="color: #999;">Attribution analysis was not computed for this protein.</p>
        </div>
    `;

    container.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Get t-SNE plot divs by identifying scatter plots
function getTSNEDivs() {
    if (!tsne2DDiv || !tsne3DDiv) {
        const allDivs = document.querySelectorAll('.plotly-graph-div');

        // Find divs that contain scatter or scatter3d traces
        let scatterDivs = [];

        for (let i = 0; i < allDivs.length; i++) {
            const div = allDivs[i];
            if (div.data && div.data.length > 0) {
                // Check if this div has scatter/scatter3d traces
                const hasScatter = div.data.some(trace =>
                    trace.type === 'scatter' ||
                    trace.type === 'scattergl' ||
                    trace.type === 'scatter3d'
                );

                if (hasScatter) {
                    scatterDivs.push(div);
                    console.log(`Found scatter plot at index ${i}, type: ${div.data[0].type}`);
                }
            }
        }

        // Assign 2D (first scatter) and 3D (first scatter3d or second scatter)
        if (scatterDivs.length > 0) {
            // Find 2D t-SNE (scatter, not scatter3d)
            tsne2DDiv = scatterDivs.find(div =>
                div.data.some(trace => trace.type === 'scatter' || trace.type === 'scattergl')
            );

            // Find 3D t-SNE (scatter3d)
            tsne3DDiv = scatterDivs.find(div =>
                div.data.some(trace => trace.type === 'scatter3d')
            );

            console.log('2D t-SNE div found:', !!tsne2DDiv);
            console.log('3D t-SNE div found:', !!tsne3DDiv);
        }
    }
    return { tsne2DDiv, tsne3DDiv };
}

// DEBUG: Log structure of data
function debugPlotData(plotDiv, plotName) {
    console.log(`=== DEBUG ${plotName} ===`);
    console.log('Number of traces:', plotDiv.data.length);

    plotDiv.data.forEach((trace, idx) => {
        console.log(`\nTrace ${idx}:`);
        console.log('  Name:', trace.name);
        console.log('  Type:', trace.type);
        console.log('  Number of points:', trace.x ? trace.x.length : 0);

        if (trace.customdata) {
            console.log('  customdata structure:', {
                isArray: Array.isArray(trace.customdata),
                length: trace.customdata.length,
                firstItem: trace.customdata[0],
                firstItemType: typeof trace.customdata[0],
                sample: trace.customdata.slice(0, 3)
            });
        } else {
            console.log('  customdata: MISSING');
        }

        if (trace.text) {
            if (Array.isArray(trace.text)) {
                console.log('  text sample:', trace.text.slice(0, 2));
            } else {
                console.log('  text:', trace.text);
            }
        }
    });
}

// Search protein in 2D plot 
function searchProtein2D() {
    const searchTerm = document.getElementById('search-2d').value.trim().toLowerCase();
    const resultSpan = document.getElementById('search-result-2d');

    if (!searchTerm) {
        resultSpan.textContent = 'Please enter a protein label';
        resultSpan.className = 'search-result error';
        return;
    }

    const { tsne2DDiv } = getTSNEDivs();
    if (!tsne2DDiv) {
        resultSpan.textContent = 'Plot not found';
        resultSpan.className = 'search-result error';
        return;
    }

    // Store original data if not done yet
    if (!originalData2D) {
        originalData2D = JSON.parse(JSON.stringify(tsne2DDiv.data));
        debugPlotData(tsne2DDiv, '2D t-SNE');
    }

    let found = false;
    let foundInTrace = -1;
    let foundAtIndex = -1;
    const newData = JSON.parse(JSON.stringify(originalData2D));

    // Search through ALL traces
    for (let traceIdx = 0; traceIdx < newData.length; traceIdx++) {
        const trace = newData[traceIdx];

        if (!trace.customdata || !Array.isArray(trace.customdata)) {
            continue;
        }

        for (let i = 0; i < trace.customdata.length; i++) {
            const item = trace.customdata[i];
            let proteinId = '';

            if (Array.isArray(item)) {
                proteinId = String(item[0]).toLowerCase();
            } else if (typeof item === 'object' && item !== null) {
                proteinId = String(item.label || item[0] || '').toLowerCase();
            } else {
                proteinId = String(item).toLowerCase();
            }

            if (proteinId.includes(searchTerm)) {
                found = true;
                foundInTrace = traceIdx;
                foundAtIndex = i;
                console.log(`FOUND: ${proteinId} at trace ${traceIdx}, index ${i}`);
                break;
            }
        }

        if (found) break;
    }

    if (found) {
        const trace = newData[foundInTrace];
        const numPoints = trace.x.length;

        // Ensure marker object exists
        if (!trace.marker) trace.marker = {};

        // Get base properties
        const baseSize = Array.isArray(trace.marker.size) ? trace.marker.size[0] : (trace.marker.size || 6);
        const baseColor = trace.marker.color;

        // Initialize arrays with base values
        trace.marker.size = new Array(numPoints);
        trace.marker.color = new Array(numPoints);
        
        for (let i = 0; i < numPoints; i++) {
            trace.marker.size[i] = typeof baseSize === 'number' ? baseSize : 6;
            
            if (Array.isArray(baseColor)) {
                trace.marker.color[i] = baseColor[i] || '#3498db';
            } else if (typeof baseColor === 'string') {
                trace.marker.color[i] = baseColor;
            } else {
                trace.marker.color[i] = '#3498db';
            }
        }

        // Apply highlight to found point
        trace.marker.size[foundAtIndex] = 25; // Augmenté de 20 à 25
        trace.marker.color[foundAtIndex] = '#FFD700'; // Gold
        
        // Add border to highlighted point
        trace.marker.line = {
            width: new Array(numPoints).fill(0),
            color: new Array(numPoints).fill('rgba(0,0,0,0)')
        };
        trace.marker.line.width[foundAtIndex] = 3; // Augmenté de 2 à 3
        trace.marker.line.color[foundAtIndex] = '#000000'; // Noir pur
        
        // Force scatter mode (not scattergl)
        trace.type = 'scatter';
        trace.mode = 'markers';
        
        // Ensure opacity is set
        if (!trace.marker.opacity) {
            trace.marker.opacity = 1;
        }

        console.log('Highlighting point:', {
            trace: foundInTrace,
            index: foundAtIndex,
            size: trace.marker.size[foundAtIndex],
            color: trace.marker.color[foundAtIndex]
        });

        Plotly.react(tsne2DDiv, newData, tsne2DDiv.layout);
        resultSpan.textContent = `Protein found in ${newData[foundInTrace].name}!`;
        resultSpan.className = 'search-result success';
    } else {
        resultSpan.textContent = 'Protein not found';
        resultSpan.className = 'search-result error';
    }
}

// Reset 2D plot highlight
function resetHighlight2D() {
    const { tsne2DDiv } = getTSNEDivs();
    const resultSpan = document.getElementById('search-result-2d');

    if (tsne2DDiv && originalData2D) {
        Plotly.react(tsne2DDiv, originalData2D, tsne2DDiv.layout);
        resultSpan.textContent = '';
        resultSpan.className = 'search-result';
        document.getElementById('search-2d').value = '';
    }
}

// Search protein in 3D plot
function searchProtein3D() {
    const searchTerm = document.getElementById('search-3d').value.trim().toLowerCase();
    const resultSpan = document.getElementById('search-result-3d');

    if (!searchTerm) {
        resultSpan.textContent = 'Please enter a protein label';
        resultSpan.className = 'search-result error';
        return;
    }

    const { tsne3DDiv } = getTSNEDivs();
    if (!tsne3DDiv) {
        resultSpan.textContent = 'Plot not found';
        resultSpan.className = 'search-result error';
        return;
    }

    if (!originalData3D) {
        originalData3D = JSON.parse(JSON.stringify(tsne3DDiv.data));
        debugPlotData(tsne3DDiv, '3D t-SNE');
    }

    let found = false;
    let foundInTrace = -1;
    let foundAtIndex = -1;
    const newData = JSON.parse(JSON.stringify(originalData3D));

    for (let traceIdx = 0; traceIdx < newData.length; traceIdx++) {
        const trace = newData[traceIdx];

        if (!trace.customdata || !Array.isArray(trace.customdata)) {
            continue;
        }

        for (let i = 0; i < trace.customdata.length; i++) {
            const item = trace.customdata[i];
            let proteinId = '';

            if (Array.isArray(item)) {
                proteinId = String(item[0]).toLowerCase();
            } else if (typeof item === 'object' && item !== null) {
                proteinId = String(item.label || item[0] || '').toLowerCase();
            } else {
                proteinId = String(item).toLowerCase();
            }

            if (proteinId.includes(searchTerm)) {
                found = true;
                foundInTrace = traceIdx;
                foundAtIndex = i;
                console.log(`FOUND 3D: ${proteinId} at trace ${traceIdx}, index ${i}`);
                break;
            }
        }

        if (found) break;
    }

    if (found) {
        const trace = newData[foundInTrace];
        const numPoints = trace.x.length;

        if (!trace.marker) trace.marker = {};

        const baseSize = Array.isArray(trace.marker.size) ? trace.marker.size[0] : (trace.marker.size || 6);
        const baseColor = trace.marker.color;

        trace.marker.size = new Array(numPoints);
        trace.marker.color = new Array(numPoints);
        
        for (let i = 0; i < numPoints; i++) {
            trace.marker.size[i] = typeof baseSize === 'number' ? baseSize : 6;
            
            if (Array.isArray(baseColor)) {
                trace.marker.color[i] = baseColor[i] || '#3498db';
            } else if (typeof baseColor === 'string') {
                trace.marker.color[i] = baseColor;
            } else {
                trace.marker.color[i] = '#3498db';
            }
        }

        trace.marker.size[foundAtIndex] = 25;
        trace.marker.color[foundAtIndex] = '#FFD700';
        
        trace.marker.line = {
            width: new Array(numPoints).fill(0),
            color: new Array(numPoints).fill('rgba(0,0,0,0)')
        };
        trace.marker.line.width[foundAtIndex] = 3;
        trace.marker.line.color[foundAtIndex] = '#000000';
        
        // Important: garder scatter3d pour la 3D
        trace.type = 'scatter3d';
        trace.mode = 'markers';
        
        if (!trace.marker.opacity) {
            trace.marker.opacity = 1;
        }

        console.log('Highlighting 3D point:', {
            trace: foundInTrace,
            index: foundAtIndex,
            size: trace.marker.size[foundAtIndex],
            color: trace.marker.color[foundAtIndex]
        });

        Plotly.react(tsne3DDiv, newData, tsne3DDiv.layout);
        resultSpan.textContent = `Protein found in ${newData[foundInTrace].name}!`;
        resultSpan.className = 'search-result success';
    } else {
        resultSpan.textContent = 'Protein not found';
        resultSpan.className = 'search-result error';
    }
}

// Reset 3D plot highlight
function resetHighlight3D() {
    const { tsne3DDiv } = getTSNEDivs();
    const resultSpan = document.getElementById('search-result-3d');

    if (tsne3DDiv && originalData3D) {
        Plotly.react(tsne3DDiv, originalData3D, tsne3DDiv.layout);
        resultSpan.textContent = '';
        resultSpan.className = 'search-result';
        document.getElementById('search-3d').value = '';
    }
}

// Protein attribution selection from dropdown
function selectProteinFromDropdown() {
    const selector = document.getElementById('protein-selector');
    const proteinId = selector.value;

    if (proteinId && attributionData[proteinId]) {
        displayAttribution(proteinId);
    }
}

// Clear attribution display
function clearAttribution() {
    const container = document.getElementById('attribution-container');
    const selector = document.getElementById('protein-selector');

    if (container) {
        container.innerHTML = `
            <p style="text-align: center; color: #999; padding: 40px;">
                Select a protein from the dropdown above
            </p>
        `;
    }

    if (selector) {
        selector.value = '';
    }
}

// Enable Enter key to trigger search
document.addEventListener('DOMContentLoaded', function() {
    const search2d = document.getElementById('search-2d');
    const search3d = document.getElementById('search-3d');

    if (search2d) {
        search2d.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') searchProtein2D();
        });
    }

    if (search3d) {
        search3d.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') searchProtein3D();
        });
    }
});