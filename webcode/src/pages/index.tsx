import { useEffect, useRef, useState, Fragment } from 'react';
import { Slider } from '@/components/ui/slider';
import defaultModel from '../models/defaultModel.json';
import model1 from '../models/model1.json';
import model2 from '../models/model2.json';
import model3 from '../models/model3.json';

// Dynamic architecture that can be updated based on loaded model
let ARCHITECTURE = {
  input: 64,
  hidden1: 10,
  hidden2: 10,
  hidden3: 10, // Added third hidden layer
  output: 4 // Default value
};

// Types for models used in UI
type ModelData = { weights: any; classes: string[]; description: string };
type AvailableModel = { name: string; data: ModelData };

const toModelData = (m: any): ModelData => ({
  weights: m.weights,
  classes: Array.isArray(m.classes) ? m.classes.map((c: any) => String(c)) : defaultClassLabels(m.weights?.output?.length || 0),
  description: m.description || ""
});

// Parse flattened image data from C array format
const parseFlattenedImage = (imageContent: string): number[] | null => {
  try {
    console.log('Parsing flattened image...');
    
    // Detect data type from the declaration
    const isUint8 = imageContent.includes('uint8_t') || imageContent.includes('unsigned char');
    const isInt8 = imageContent.includes('int8_t') || imageContent.includes('signed char');
    const isInt = imageContent.includes('int ') && !isInt8;
    
    console.log(`Detected data type: ${isUint8 ? 'uint8_t' : isInt8 ? 'int8_t' : isInt ? 'int' : 'unknown'}`);
    
    // Extract array values using regex
    const arrayMatch = imageContent.match(/\{([^}]+)\}/);
    if (!arrayMatch) {
      console.error('No array found in image content');
      return null;
    }
    
    const valuesString = arrayMatch[1];
    const values = valuesString.split(',').map(v => {
      const trimmed = v.trim();
      // Handle different number formats (int8_t, uint8_t, int, hex, etc.)
      if (trimmed.startsWith('0x')) {
        return parseInt(trimmed, 16);
      }
      return parseInt(trimmed, 10);
    });
    
    // Validate array length (should be 784 for 28x28 MNIST images)
    if (values.length !== 784) {
      console.warn(`Expected 784 values, got ${values.length}`);
    }
    
    // Normalize values to 0-1 range based on detected data type
    const normalizedValues = values.map(v => {
      if (isUint8) {
        // uint8_t: 0 to 255 -> 0 to 1
        return v / 255;
      } else if (isInt8) {
        // int8_t: -128 to 127 -> 0 to 1
        return Math.max(0, (v + 128) / 255);
      } else if (isInt) {
        // int: assume 0 to 255 range, clamp to 0-1
        return Math.max(0, Math.min(1, v / 255));
      } else {
        // Default: assume 0-255 range
        return Math.max(0, Math.min(1, v / 255));
      }
    });
    
    console.log(`Parsed image with ${values.length} values, data type: ${isUint8 ? 'uint8_t' : isInt8 ? 'int8_t' : 'int'}`);
    return normalizedValues;
  } catch (error) {
    console.error('Error parsing flattened image:', error);
    return null;
  }
};

// Parse quantized model from C header file
const parseQuantizedModel = (headerContent: string) => {
  try {
    console.log('Parsing quantized model...');
    const lines = headerContent.split('\n');
    const model: any = { layers: [], weights: {} };
    
    let currentLayer: any = null;
    
    for (const line of lines) {
      const trimmedLine = line.trim();
      
      // Parse layer definitions
      if (trimmedLine.startsWith('#define L') && trimmedLine.includes('_active')) {
        const layerName = trimmedLine.split(' ')[1];
        currentLayer = { name: layerName, weights: [] };
        model.layers.push(currentLayer);
        console.log(`Found layer: ${layerName}`);
      }
      
      // Parse layer properties
      if (trimmedLine.startsWith('#define L') && trimmedLine.includes('_incoming_weights')) {
        const match = trimmedLine.match(/_incoming_weights (\d+)/);
        if (match && currentLayer) {
          currentLayer.incoming = parseInt(match[1]);
          console.log(`  Input size: ${currentLayer.incoming}`);
        }
      }
      
      if (trimmedLine.startsWith('#define L') && trimmedLine.includes('_outgoing_weights')) {
        const match = trimmedLine.match(/_outgoing_weights (\d+)/);
        if (match && currentLayer) {
          currentLayer.outgoing = parseInt(match[1]);
          console.log(`  Output size: ${currentLayer.outgoing}`);
        }
      }
      
      if (trimmedLine.startsWith('#define L') && trimmedLine.includes('_bitperweight')) {
        const match = trimmedLine.match(/_bitperweight (\d+)/);
        if (match && currentLayer) {
          currentLayer.bits = parseInt(match[1]);
          console.log(`  Bits per weight: ${currentLayer.bits}`);
        }
      }
      
      // Parse weights array
      if (trimmedLine.startsWith('const uint32_t L') && trimmedLine.includes('_weights[]')) {
        const layerMatch = trimmedLine.match(/L(\d+)_weights/);
        if (layerMatch) {
          const layerIndex = parseInt(layerMatch[1]) - 1;
          if (model.layers[layerIndex]) {
            model.layers[layerIndex].weightArrayName = `L${layerIndex + 1}_weights`;
          }
        }
      }
      
      // Parse weight values
      if (trimmedLine.includes('0x') && trimmedLine.includes(',')) {
        const hexValues = trimmedLine.match(/0x[0-9a-fA-F]+/g);
        if (hexValues && currentLayer) {
          currentLayer.weights.push(...hexValues);
        }
      }
    }
    
    console.log(`Parsed ${model.layers.length} layers`);
    
    // Convert to standard format
    const standardWeights: any = {};
    let inputSize = 784; // Default for MNIST
    
    for (let i = 0; i < model.layers.length; i++) {
      const layer = model.layers[i];
      if (!layer.incoming || !layer.outgoing) {
        console.warn(`Layer ${i} missing incoming or outgoing weights`);
        continue;
      }
      
      // Determine layer name based on position and total layers
      let layerName;
      if (model.layers.length === 4) {
        // 3 hidden layers + output
        layerName = i === 0 ? 'hidden1' : 
                   i === 1 ? 'hidden2' : 
                   i === 2 ? 'hidden3' : 'output';
      } else if (model.layers.length === 3) {
        // 2 hidden layers + output
        layerName = i === 0 ? 'hidden1' : 
                   i === 1 ? 'hidden2' : 'output';
      } else {
        // Generic naming for other cases
        layerName = i === 0 ? 'hidden1' : 
                   i === 1 ? 'hidden2' : 
                   i === 2 ? 'hidden3' : `layer${i + 1}`;
      }
      
      console.log(`Processing ${layerName}: ${layer.incoming} -> ${layer.outgoing} (${layer.bits} bits)`);
      
      // Extract weights from hex values based on quantization
      const weights = extractQuantizedWeights(layer.weights, layer.bits, layer.incoming, layer.outgoing);
      
      standardWeights[layerName] = weights;
      
      if (i === 0) {
        inputSize = layer.incoming;
      }
    }
    
    // Determine architecture based on number of layers
    const numLayers = model.layers.length;
    const architecture = {
      input: inputSize,
      hidden1: model.layers[0]?.outgoing || 16,
      hidden2: model.layers[1]?.outgoing || 16,
      hidden3: numLayers === 4 ? (model.layers[2]?.outgoing || 16) : 0, // Only set if 3 hidden layers
      output: numLayers === 4 ? (model.layers[3]?.outgoing || 10) : (model.layers[2]?.outgoing || 10)
    };
    
    const result = {
      weights: standardWeights,
      architecture,
      classes: defaultClassLabels(architecture.output),
      description: `Quantized model loaded from C header file (${numLayers} layers)`
    };
    
    console.log('Parsed model:', result);
    return result;
  } catch (error) {
    console.error('Error parsing quantized model:', error);
    return null;
  }
};

// Extract weights from quantized hex values
const extractQuantizedWeights = (hexWeights: string[], bits: number, incoming: number, outgoing: number): number[][] => {
  const weights: number[][] = [];
  
  // Convert hex strings to binary and extract weights
  for (let out = 0; out < outgoing; out++) {
    // console.log("out", out, outgoing, incoming);
    weights[out] = [];
    for (let in_ = 0; in_ < incoming; in_++) {
      // console.log("in_", in_);
      const weightIndex = Math.floor((out * incoming + in_) / (32 / bits));
      const bitOffset = (out * incoming + in_) % (32 / bits);
      
      if (weightIndex < hexWeights.length) {
        const hexValue = parseInt(hexWeights[weightIndex], 16);
        const weight = extractWeightFromBits(hexValue, bitOffset, bits);
        weights[out][in_] = weight;
      } else {
        weights[out][in_] = 0;
      }
    }
  }
  
  return weights;
};

// Extract weight value from bits at specific position
const extractWeightFromBits = (hexValue: number, bitOffset: number, bits: number): number => {
  const mask = (1 << bits) - 1;
  const weight = (hexValue >> (bitOffset * bits)) & mask;
  
  // Convert to signed value (assuming symmetric quantization)
  const maxValue = (1 << (bits - 1)) - 1;
  if (weight > maxValue) {
    return weight - (1 << bits);
  }
  return weight;
};

// Update global architecture
const updateArchitecture = (newArchitecture: any) => {
  ARCHITECTURE = { ...newArchitecture };
};

// Default class labels generator: ["0", "1", ..., String(n-1)]
const defaultClassLabels = (n: number): string[] => Array.from({ length: n }, (_, i) => String(i));

// Helper to multiply matrix and vector
const matrixVectorProduct = (matrix: number[][], vector: number[]): number[] => {
  return matrix.map(row => 
    row.reduce((sum, weight, i) => sum + weight * vector[i], 0)
  );
};

// ReLU activation function
const relu = (x: number): number => Math.max(0, x);

// Layer normalization (mean=0, variance=1)
const layerNorm = (vector: number[]): number[] => {
  // Calculate mean
  const mean = vector.reduce((sum, x) => sum + x, 0) / vector.length;
  
  // Calculate variance
  const variance = vector.reduce((sum, x) => sum + (x - mean) * (x - mean), 0) / vector.length;
  
  // Add epsilon for numerical stability
  const std = Math.sqrt(variance + 1e-5);
  
  // Normalize
  return vector.map(x => (x - mean) / std);
};

// Center the input pattern
const centerInputPattern = (input: number[]): number[] => {
  const gridSize = Math.sqrt(input.length);
  if (!Number.isInteger(gridSize)) {
    // If not a perfect square, return input as-is
    return input;
  }
  
  const nonZeroIndices = input
    .map((value, index) => (value > 0 ? index : -1))
    .filter(index => index !== -1);

  if (nonZeroIndices.length === 0) return input;

  const xCoords = nonZeroIndices.map(index => index % gridSize);
  const yCoords = nonZeroIndices.map(index => Math.floor(index / gridSize));

  const xCenter = (Math.min(...xCoords) + Math.max(...xCoords)) / 2;
  const yCenter = (Math.min(...yCoords) + Math.max(...yCoords)) / 2;

  const xOffset = Math.floor(gridSize / 2) - Math.round(xCenter);
  const yOffset = Math.floor(gridSize / 2) - Math.round(yCenter);

  const centeredInput = new Array(input.length).fill(0);
  nonZeroIndices.forEach(index => {
    const x = index % gridSize;
    const y = Math.floor(index / gridSize);
    const newX = x + xOffset;
    const newY = y + yOffset;
    if (newX >= 0 && newX < gridSize && newY >= 0 && newY < gridSize) {
      centeredInput[newY * gridSize + newX] = input[index];
    }
  });

  return centeredInput;
};

// Modify the forwardPass function to accept centerInput as a parameter:
const forwardPass = (input: number[], weights: any, centerInput: boolean): any => {
  // Resize input to match the expected input size
  let processedInput = input;
  if (input.length !== ARCHITECTURE.input) {
    if (input.length < ARCHITECTURE.input) {
      // Pad with zeros if input is too small
      processedInput = [...input, ...new Array(ARCHITECTURE.input - input.length).fill(0)];
    } else {
      // Truncate if input is too large
      processedInput = input.slice(0, ARCHITECTURE.input);
    }
  }
  
  processedInput = centerInput ? centerInputPattern(processedInput) : processedInput;
  
  // First block
  const norm1 = layerNorm(processedInput);
  const linear1 = matrixVectorProduct(weights.hidden1, norm1);
  const act1 = linear1.map(relu);
  
  // Second block
  const norm2 = layerNorm(act1);
  const linear2 = matrixVectorProduct(weights.hidden2, norm2);
  const act2 = linear2.map(relu);
  
  let act3, output;
  
  // Check if we have a third hidden layer
  if (weights.hidden3 && ARCHITECTURE.hidden3 > 0) {
    // Third block (3 hidden layers)
    const norm3 = layerNorm(act2);
    const linear3 = matrixVectorProduct(weights.hidden3, norm3);
    act3 = linear3.map(relu);
  
  // Output block
    const norm4 = layerNorm(act3);
    output = matrixVectorProduct(weights.output, norm4);
  } else {
    // Output block (2 hidden layers)
  const norm3 = layerNorm(act2);
    output = matrixVectorProduct(weights.output, norm3);
  }

  const result: any = {
    hidden1: act1,      // Post-ReLU
    hidden2: act2,      // Post-ReLU
    output: output      // Post-Linear (no ReLU)
  };
  
  // Only add hidden3 if it exists
  if (weights.hidden3 && ARCHITECTURE.hidden3 > 0) {
    result.hidden3 = act3;
  }
  
  return result;
};



const FuelGauge = ({ value, minValue, maxValue, isHighest = false }: { value: number, minValue: number, maxValue: number, isHighest?: boolean }) => {
  const height = 40;
  const width = 10;
  
  // Scale value between 0 and 1
  const scaledValue = maxValue === minValue ? 
    (maxValue === 0 ? 0 : 0.5) : // If all zeros show empty, otherwise show half
    (value - minValue) / (maxValue - minValue);
  
  const fillHeight = height * scaledValue;
  
  return (
    <svg width={width} height={height}>
      <rect
        x={1}
        y={0}
        width={width - 2}
        height={height}
        fill="#333"
        rx={2}
      />
      <rect
        x={2}
        y={height - fillHeight}
        width={width - 4}
        height={fillHeight}
        fill={isHighest ? '#FF009E' : '#00E5FF'}
        rx={1}
      />
    </svg>
  );
};

  // Add this at the top level of the file, after the imports
const validateWeights = (weights: any): boolean => {
  if (!weights || typeof weights !== 'object') return false;
  
  // Check structure - require at least hidden1, hidden2, and output
  const required = ['hidden1', 'hidden2', 'output'];
  if (!required.every(key => key in weights)) return false;
  
  // Check if we have a third hidden layer
  const hasHidden3 = 'hidden3' in weights && weights.hidden3;
  
  // Check dimensions
  try {
    if (weights.hidden1.length !== ARCHITECTURE.hidden1) return false;
    if (weights.hidden1[0].length !== ARCHITECTURE.input) return false;
    
    if (weights.hidden2.length !== ARCHITECTURE.hidden2) return false;
    if (weights.hidden2[0].length !== ARCHITECTURE.hidden1) return false;
    
    if (hasHidden3) {
      if (weights.hidden3.length !== ARCHITECTURE.hidden3) return false;
      if (weights.hidden3[0].length !== ARCHITECTURE.hidden2) return false;
      
      if (weights.output.length > 10 || weights.output.length < 1) return false;
      if (weights.output[0].length !== ARCHITECTURE.hidden3) return false;
    } else {
      if (weights.output.length > 10 || weights.output.length < 1) return false;
    if (weights.output[0].length !== ARCHITECTURE.hidden2) return false;
    }
    
    // Check if all values are numbers
    const allNumbers = (arr: any[]): boolean => arr.flat().every(x => typeof x === 'number' && !isNaN(x));
    if (!allNumbers(weights.hidden1) || !allNumbers(weights.hidden2) || !allNumbers(weights.output)) {
      return false;
    }
    
    if (hasHidden3 && !allNumbers(weights.hidden3)) {
      return false;
    }

    // Classes array is optional
    if (weights.classes && (!Array.isArray(weights.classes) || weights.classes.length !== weights.output.length)) {
      return false;
    }
  } catch (e) {
    return false;
  }
  
  return true;
};

const NetworkViz = (
  { activations, config, centerInput, setCenterInput }: 
  { 
    activations: number[], 
    config: any, 
    centerInput: boolean, 
    setCenterInput: React.Dispatch<React.SetStateAction<boolean>> 
  }
) => {
  const [weightThreshold, setWeightThreshold] = useState(50);
  const [weights, setWeights] = useState(config.weights);
  const [classLabels, setClassLabels] = useState(config.classes || defaultClassLabels(config.weights.output.length));
  const [layerActivations, setLayerActivations] = useState<{ hidden1: number[], hidden2: number[], hidden3?: number[], output: number[] } | null>(null);

  // Update weights and class labels when config changes
  useEffect(() => {
    setWeights(config.weights);
    setClassLabels(config.classes || defaultClassLabels(config.weights.output.length));
  }, [config]);
  
  // Update architecture when it changes
  useEffect(() => {
    // This will trigger a re-render with the new architecture
  }, [ARCHITECTURE]);
  
  useEffect(() => {
    if (activations) {
      setLayerActivations(forwardPass(activations, weights, centerInput));
    }
  }, [activations, weights, centerInput]);

  const getLayerPositions = (layerSize: number, layerIndex: number, totalLayers: number): { x: number, y: number }[] => {
    const width = 900;  // Increased by 50%
    const height = 450; // Increased by 50%
    const padding = 40; // Padding for fuel gauges
    
    // Adjust x position to account for padding
    const layerX = padding + ((width - 2 * padding) / (totalLayers - 1)) * layerIndex;
    const positions = [];
    
    // Calculate vertical spacing based on layer size
    const verticalPadding = 30;
    const availableHeight = height - 2 * verticalPadding;
    const spacing = availableHeight / (layerSize - 1);
    
    for (let i = 0; i < layerSize; i++) {
      const y = verticalPadding + (spacing * i);
      positions.push({ x: layerX, y });
    }
    return positions;
  };

  const renderConnections = () => {
    // Get current architecture dynamically
    const currentArchitecture = {
      input: ARCHITECTURE.input,
      hidden1: weights?.hidden1?.length || ARCHITECTURE.hidden1,
      hidden2: weights?.hidden2?.length || ARCHITECTURE.hidden2,
      hidden3: weights?.hidden3?.length || ARCHITECTURE.hidden3, // Added hidden3
      output: weights?.output?.length || ARCHITECTURE.output
    };
    
    // Determine layers based on whether hidden3 exists
    const hasHidden3 = weights?.hidden3 && currentArchitecture.hidden3 > 0;
    const layers = hasHidden3 
      ? [currentArchitecture.input, currentArchitecture.hidden1, currentArchitecture.hidden2, currentArchitecture.hidden3, currentArchitecture.output]
      : [currentArchitecture.input, currentArchitecture.hidden1, currentArchitecture.hidden2, currentArchitecture.output];
    
    const allPositions = layers.map((size, i) => getLayerPositions(size, i, layers.length));
    const connections: JSX.Element[] = [];
    let connectionId = 0;

    // Get source activations for each layer
    const getLayerActivations = (layerIndex: number): number[] => {
      if (!layerActivations) return new Array(layers[layerIndex]).fill(0);
      switch(layerIndex) {
        case 0: return centerInput ? centerInputPattern(activations || new Array(ARCHITECTURE.input).fill(0)) : (activations || new Array(ARCHITECTURE.input).fill(0));
        case 1: return layerActivations.hidden1;
        case 2: return layerActivations.hidden2;
        case 3: return hasHidden3 ? (layerActivations.hidden3 || []) : layerActivations.output;
        case 4: return hasHidden3 ? layerActivations.output : [];
        default: return [];
      }
    };

    const getVisibleProducts = (weights: number[][], sourceActs: number[]): { threshold: number, maxProduct: number } => {
      // Calculate all activation*weight products
      const products = weights.map(row => 
        row.map((w, j) => w * sourceActs[j])
      ).flat();
      
      const allProducts = products.map(Math.abs);
      const sorted = [...allProducts].sort((a, b) => a - b);
      const threshold = sorted[Math.floor((sorted.length - 1) * (weightThreshold / 100))];
      return { threshold, maxProduct: sorted[sorted.length - 1] };
    };

    const drawLayerConnections = (layer1Pos: { x: number, y: number }[], layer2Pos: { x: number, y: number }[], weights: number[][], startIndex: number) => {
      const sourceActs = getLayerActivations(startIndex);
      const { threshold, maxProduct } = getVisibleProducts(weights, sourceActs);
      
      layer1Pos.forEach((start, i) => {
        layer2Pos.forEach((end, j) => {
          const weight = weights[j][i];
          const product = weight * sourceActs[i];
          const absProduct = Math.abs(product);
          
          // Only draw connection if product is non-zero and above threshold
          if (absProduct > 0 && absProduct >= threshold) {
            // Scale thickness relative to visible products
            const scaledIntensity = (absProduct - threshold) / (maxProduct - threshold);
            const strokeWidth = 0.5 + (scaledIntensity * 2);
            const color = product > 0 ? '#00E5FF' : '#FF9E00';  // Swapped colors
            
            connections.push(
              <line
                key={`conn-${startIndex}-${connectionId++}`}
                x1={start.x}
                y1={start.y}
                x2={end.x}
                y2={end.y}
                stroke={color}
                strokeWidth={strokeWidth}
                opacity={0.5}
              />
            );
          }
        });
      });
    };

    if (weights) {
      drawLayerConnections(allPositions[0], allPositions[1], weights.hidden1, 0);
      drawLayerConnections(allPositions[1], allPositions[2], weights.hidden2, 1);
      
      if (hasHidden3) {
        drawLayerConnections(allPositions[2], allPositions[3], weights.hidden3, 2);
        drawLayerConnections(allPositions[3], allPositions[4], weights.output, 3);
      } else {
      drawLayerConnections(allPositions[2], allPositions[3], weights.output, 2);
      }
    }

    return connections;
  };

  const renderNeurons = () => {
    // Get current architecture dynamically
    const currentArchitecture = {
      input: ARCHITECTURE.input,
      hidden1: weights?.hidden1?.length || ARCHITECTURE.hidden1,
      hidden2: weights?.hidden2?.length || ARCHITECTURE.hidden2,
      hidden3: weights?.hidden3?.length || ARCHITECTURE.hidden3, // Added hidden3
      output: weights?.output?.length || ARCHITECTURE.output
    };
    
    // Determine layers based on whether hidden3 exists
    const hasHidden3 = weights?.hidden3 && currentArchitecture.hidden3 > 0;
    const layers = hasHidden3 
      ? [currentArchitecture.input, currentArchitecture.hidden1, currentArchitecture.hidden2, currentArchitecture.hidden3, currentArchitecture.output]
      : [currentArchitecture.input, currentArchitecture.hidden1, currentArchitecture.hidden2, currentArchitecture.output];
    
    const neurons: JSX.Element[] = [];
    let neuronId = 0;

    layers.forEach((layerSize, layerIndex) => {
      const positions = getLayerPositions(layerSize, layerIndex, layers.length);
      positions.forEach((pos, i) => {
        // Get activation value for this neuron
        let activation = 0;
        if (layerIndex === 0) {
          activation = centerInput ? centerInputPattern(activations || new Array(ARCHITECTURE.input).fill(0))[i] : (activations || new Array(ARCHITECTURE.input).fill(0))[i];
        } else if (layerActivations) {
          const layerName = layerIndex === 1 ? 'hidden1' : 
                           layerIndex === 2 ? 'hidden2' : 
                           layerIndex === 3 ? (hasHidden3 ? 'hidden3' : 'output') : 'output';
          activation = layerActivations[layerName]?.[i] || 0;
        }

        // For output layer, determine if this is the highest activation
        const isOutputLayer = hasHidden3 ? layerIndex === 4 : layerIndex === 3;
        const isHighestOutput = isOutputLayer && layerActivations?.output && 
          activation === Math.max(...layerActivations.output);

        // Draw neuron
        neurons.push(
          <g key={`neuron-${neuronId++}`}>
            {layerIndex === 0 ? (
              // Square with outline for input layer
              <rect
                x={pos.x - 4}
                y={pos.y - 4}
                width={8}
                height={8}
                fill={`rgba(255, 255, 255, ${activation})`}
                stroke="white"
                strokeWidth="1"
              />
            ) : (
              // Circle for other layers
              <circle
                cx={pos.x}
                cy={pos.y}
                r={4}
                fill="#FFFFFF"
                opacity={0.8}
              />
            )}
            {layerIndex > 0 && (
              <foreignObject
                x={pos.x + 8}
                y={pos.y - 20}
                width={isOutputLayer ? 50 : 12}
                height={40}
                className="flex items-center"
              >
                <div className="flex items-center gap-2">
                  <FuelGauge 
                    value={activation || 0} 
                    minValue={layerIndex === 0 ? 0 : 
                      !layerActivations ? 0 : Math.min(...(
                        layerIndex === 1 ? layerActivations.hidden1 :
                        layerIndex === 2 ? layerActivations.hidden2 :
                        layerIndex === 3 ? (hasHidden3 ? (layerActivations.hidden3 || []) : layerActivations.output) :
                        layerActivations.output
                      ))}
                    maxValue={layerIndex === 0 ? 1 : 
                      !layerActivations ? 1 : Math.max(...(
                        layerIndex === 1 ? layerActivations.hidden1 :
                        layerIndex === 2 ? layerActivations.hidden2 :
                        layerIndex === 3 ? (hasHidden3 ? (layerActivations.hidden3 || []) : layerActivations.output) :
                        layerActivations.output
                      ))}
                    isHighest={isHighestOutput}
                  />
                  {isOutputLayer && (
                    <span className="text-xs text-white ml-1 font-bold">
                      {classLabels[i]}
                    </span>
                  )}
                </div>
              </foreignObject>
            )}
          </g>
        );
      });
    });

    return neurons;
  };

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="flex items-center gap-4 text-xs text-gray-300 mb-2">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#00E5FF' }}></div>
          <span>Positive activation flow</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#FF9E00' }}></div>
          <span>Negative activation flow</span>
        </div>
      </div>
      <div 
        className="relative p-1 rounded-lg"
        style={{
          background: 'linear-gradient(45deg, #FF9E0011, #00E5FF11)',
          boxShadow: `
            0 0 20px 0 #FF9E0022,
            inset 0 0 20px 0 #00E5FF22
          `,
        }}
      >
        <div 
          className="absolute inset-0 rounded-lg"
          style={{
            background: 'linear-gradient(45deg, #FF9E00, #00E5FF)',
            opacity: 0.1,
            filter: 'blur(20px)',
          }}
        />
        <svg width="900" height="450" className="bg-gray-900 rounded-lg relative z-10">
          {renderConnections()}
          {renderNeurons()}
        </svg>
      </div>
      <div className="w-full mt-4 flex items-center gap-4 flex-nowrap">
        <label className="text-sm text-gray-300">Connection Threshold: {weightThreshold}%</label>
        <Slider
          value={[weightThreshold]}
          onValueChange={([value]) => setWeightThreshold(value)}
          min={0}
          max={100}
          step={1}
          className="mt-1 flex-1"
        />
        <label className="text-sm text-gray-300 flex items-center gap-2">
          <input 
            type="checkbox" 
            checked={centerInput} 
            onChange={() => setCenterInput(!centerInput)} 
          />
          Center Input
        </label>
      </div>
    </div>
  );
};

const DrawingCanvas = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const flattenedImageInputRef = useRef<HTMLTextAreaElement | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [pixelData, setPixelData] = useState(new Array(64).fill(0));
  const [canvasSize, setCanvasSize] = useState(8); // Grid size for canvas
  
  // Update canvas size when architecture changes
  useEffect(() => {
    const inputSize = ARCHITECTURE.input;
    if (inputSize === 784) {
      setCanvasSize(28); // MNIST size
      setPixelData(new Array(784).fill(0));
    } else if (inputSize === 64) {
      setCanvasSize(8); // Default size
      setPixelData(new Array(64).fill(0));
    } else {
      // For other sizes, try to find a square root
      const sqrt = Math.sqrt(inputSize);
      if (Number.isInteger(sqrt)) {
        setCanvasSize(sqrt);
        setPixelData(new Array(inputSize).fill(0));
      } else {
        // Fallback to 8x8 for non-square inputs
        setCanvasSize(8);
        setPixelData(new Array(64).fill(0));
      }
    }
  }, [ARCHITECTURE.input]);
  
  // Convert AVAILABLE_MODELS to a state variable
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([
    { name: 'Default Model', data: toModelData(defaultModel) },
    { name: 'Model 1', data: toModelData(model1) },
    { name: 'Model 2', data: toModelData(model2) },
    { name: 'Model 3', data: toModelData(model3) }
  ]);

  const [networkConfig, setNetworkConfig] = useState<ModelData>({
    weights: defaultModel.weights,
    classes: Array.isArray(defaultModel.classes) ? defaultModel.classes.map((c: any) => String(c)) : defaultClassLabels(defaultModel.weights.output.length),
    description: defaultModel.description || ""
  });
  
  const [errorMessage, setErrorMessage] = useState("");
  const [centerInput, setCenterInput] = useState(true);
  const [selectedModel, setSelectedModel] = useState('Default Model');
  const CANVAS_SIZE = 256;
  const GRID_SIZE = canvasSize;
  const PIXEL_SIZE = CANVAS_SIZE / GRID_SIZE;
  const [brushSize, setBrushSize] = useState(1);
  const [fullIntensity, setFullIntensity] = useState(false);

  // Handle weight upload and add "Custom Model" to the selection list
  const handleWeightUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files) {
      setErrorMessage("No file selected");
      return;
    }
    const file = event.target.files[0];
    if (!file) return;
    setErrorMessage(""); // Clear any previous error

    const reader = new FileReader();
    reader.onload = (e) => {
      if (!e.target?.result) {
        setErrorMessage("Error reading file");
        return;
      }
      try {
        const json = JSON.parse(e.target.result as string);
        const { weights, classes, description } = json;
        
        if (!weights) {
          setErrorMessage("Error: No weights found in file");
          return;
        }

        if (!validateWeights(weights)) {
          setErrorMessage(
            "Invalid weight format. Expected dimensions: " +
            `Input→Hidden1: 64×10, Hidden1→Hidden2: 10×10, Hidden2→Hidden3: 10×10, Hidden3→Output: 10×4`
          );
          return;
        }

        // Only use provided classes if they exist and have the correct length
        const validClasses = Array.isArray(classes) && classes.length === weights.output.length;

        setNetworkConfig({
          weights,
          classes: validClasses ? classes.map((c: any) => String(c)) : defaultClassLabels(weights.output.length),
          description: description || ""
        });

        // Add "Custom Model" to availableModels if not already present
        setAvailableModels((prevModels: AvailableModel[]) => {
          // Check if "Custom Model" already exists
          const customModelExists = prevModels.some(model => model.name === 'Custom Model');
          if (!customModelExists) {
            return [...prevModels, { name: 'Custom Model', data: { weights, classes: validClasses ? classes.map((c: any) => String(c)) : defaultClassLabels(weights.output.length), description: description || "" } }];
          } else {
            // Update the existing "Custom Model" with new data
            return prevModels.map(model => 
              model.name === 'Custom Model' ? { name: 'Custom Model', data: { weights, classes: validClasses ? classes.map((c: any) => String(c)) : defaultClassLabels(weights.output.length), description: description || "" } } : model
            );
          }
        });
      } catch (error) {
        setErrorMessage("Error parsing file. Please ensure it's valid JSON.");
      }
    };

    reader.onerror = () => {
      setErrorMessage("Error reading file");
    };

    reader.readAsText(file);
    // Reset file input
    event.target.value = '';
  };

  // Handle model selection, including "Custom Model"
  const handleModelSelect = (modelName: string) => {
    const selected = availableModels.find(model => model.name === modelName);
    if (selected) {
      setNetworkConfig({
        weights: selected.data.weights,
        classes: selected.data.classes || defaultClassLabels(selected.data.weights.output.length),
        description: selected.data.description || ""
      });
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Type assertion to satisfy TypeScript
    const context = ctx as CanvasRenderingContext2D;
    context.fillStyle = '#1A1A1A';
    context.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    
    context.strokeStyle = '#333333';
    for (let i = 1; i < GRID_SIZE; i++) {
      const pos = i * PIXEL_SIZE;
      context.beginPath();
      context.moveTo(pos, 0);
      context.lineTo(pos, CANVAS_SIZE);
      context.moveTo(0, pos);
      context.lineTo(CANVAS_SIZE, pos);
      context.stroke();
    }
  }, [GRID_SIZE, PIXEL_SIZE]);

  const updatePixelData = (ctx: CanvasRenderingContext2D, x: number, y: number, intensity: number) => {
    const centerGX = Math.floor(x / PIXEL_SIZE);
    const centerGY = Math.floor(y / PIXEL_SIZE);

      const newPixelData = [...pixelData];

    // Map brushSize: 1 -> 1x1, 2 -> 3x3, 3 -> 4x4, and so on
    const boxSize = brushSize === 1 ? 1 : brushSize + 1;
    const halfSpan = Math.floor((boxSize - 1) / 2);
    const startGX = centerGX - halfSpan;
    const startGY = centerGY - halfSpan;
    const endGX = startGX + boxSize - 1;
    const endGY = startGY + boxSize - 1;

    for (let gy = startGY; gy <= endGY; gy++) {
      if (gy < 0 || gy >= GRID_SIZE) continue;
      for (let gx = startGX; gx <= endGX; gx++) {
        if (gx < 0 || gx >= GRID_SIZE) continue;
        const index = gy * GRID_SIZE + gx;
        newPixelData[index] = fullIntensity ? 1 : Math.min(1, newPixelData[index] + intensity);

        ctx.fillStyle = `rgba(255, 255, 255, ${newPixelData[index]})`;
        ctx.fillRect(gx * PIXEL_SIZE, gy * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
      ctx.strokeStyle = '#333333';
        ctx.strokeRect(gx * PIXEL_SIZE, gy * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
    }
    }

    setPixelData(newPixelData);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    updatePixelData(ctx, x, y, 0.2);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!canvas) return;
    
    // Type assertion to satisfy TypeScript
    const context = ctx as CanvasRenderingContext2D;
    context.fillStyle = '#1A1A1A';
    context.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    
    context.strokeStyle = '#333333';
    for (let i = 0; i < GRID_SIZE; i++) {
      for (let j = 0; j < GRID_SIZE; j++) {
        context.strokeRect(i * PIXEL_SIZE, j * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
      }
    }
    
    setPixelData(new Array(ARCHITECTURE.input).fill(0));
  };

  // Load flattened image data to canvas
  const loadFlattenedImage = (imageData: number[]) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Type assertion to satisfy TypeScript
    const context = ctx as CanvasRenderingContext2D;
    
    // Clear canvas first
    context.fillStyle = '#1A1A1A';
    context.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    
    // Draw grid
    context.strokeStyle = '#333333';
    for (let i = 1; i < GRID_SIZE; i++) {
      const pos = i * PIXEL_SIZE;
      context.beginPath();
      context.moveTo(pos, 0);
      context.lineTo(pos, CANVAS_SIZE);
      context.moveTo(0, pos);
      context.lineTo(CANVAS_SIZE, pos);
      context.stroke();
    }
    
    // Update pixel data first to ensure synchronization
    const newPixelData = [...imageData];
    setPixelData(newPixelData);
    
    // Draw the image pixels
    for (let i = 0; i < GRID_SIZE; i++) {
      for (let j = 0; j < GRID_SIZE; j++) {
        const index = i * GRID_SIZE + j;
        if (index < imageData.length) {
          const intensity = imageData[index];
          if (intensity > 0) {
            context.fillStyle = `rgba(255, 255, 255, ${intensity})`;
            context.fillRect(j * PIXEL_SIZE, i * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
            context.strokeStyle = '#333333';
            context.strokeRect(j * PIXEL_SIZE, i * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
          }
        }
      }
    }
    
    // Force a re-render of the network visualization
    setTimeout(() => {
      // This will trigger the useEffect in NetworkViz to recalculate activations
      setPixelData([...newPixelData]);
    }, 0);
  };

  return (
    <div className="min-h-screen w-full bg-gray-900 flex flex-col gap-4 p-8 text-white items-center">
      <h1 
        className="text-4xl font-bold mb-4"
        style={{ 
          color: '#FF9E00',
          textShadow: `
            0 0 20px #FF9E00AA,
            0 0 40px #FF9E0088,
            0 0 60px #FF9E0044,
             2px 2px 2px rgba(0, 0, 0, 0.5)
          `
        }}
      >
        Neural Network Visualizer for RISC-V Edge AI Workshop using VSDSquadron Pro
      </h1>
      <h2>Developed by <a href="https://www.linkedin.com/in/ankitmawle/" target="_blank" rel="noopener noreferrer">Ankit Mawle</a></h2>
      <div>
        <div className="text-sm text-[#FF009E] min-w-[250px] mt-1 px-2 text-center">
          <a 
            href="https://github.com/ankitmawle/neural-network-visualizer" 
            target="_blank" 
            rel="noopener noreferrer" 
            className="flex items-center justify-center mt-1"
          >
            <img 
              src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" 
              alt="GitHub Logo" 
              className="w-4 h-4 mr-1"
            /> Link to Repository </a>
        </div>
      </div>
   
      {errorMessage && (
        <div className="px-4 py-2 bg-red-900/50 border border-red-500 rounded mb-4 text-red-200">
          {errorMessage}
        </div>
      )}
      <div className="flex gap-4 items-start justify-center">
        <div className="flex flex-col items-center">
        <canvas
          ref={canvasRef}
          width={CANVAS_SIZE}
          height={CANVAS_SIZE}
          className="border border-gray-600 cursor-crosshair"
          onMouseDown={(e) => {
            setIsDrawing(true);
            handleMouseMove(e);
          }}
          onMouseMove={handleMouseMove}
          onMouseUp={() => setIsDrawing(false)}
          onMouseLeave={() => setIsDrawing(false)}
        />
          
          {/* Canvas Controls Below Canvas */}
          <div className="flex items-center gap-4 mt-3">
            
            <div className="flex items-center gap-3">
              <label className="text-xs text-gray-300 whitespace-nowrap">Brush size: {brushSize}</label>
              <Slider
                value={[brushSize]}
                onValueChange={([v]) => setBrushSize(Math.max(1, Math.min(10, Math.round(v))))}
                min={1}
                max={10}
                step={1}
                className="mt-1 w-32"
              />
            </div>
            
            <label className="text-xs text-gray-300 flex items-center gap-2">
              <input
                type="checkbox"
                checked={fullIntensity}
                onChange={() => setFullIntensity(!fullIntensity)}
              />
              Full intensity (1.0)
            </label>
          </div>
          <div className="flex items-center gap-4 mt-3">
          <button
            onClick={clearCanvas}
              className="px-4 py-2 bg-gray-800 text-[#00E5FF] border border-[#00E5FF] rounded hover:bg-[#00E5FF22] transition-colors font-medium"
          >
            Clear
          </button>
     
          
          </div>
          
          {/* genrate image data from canvas */}
          <div className="mt-3">
              {/* Flattened Image Input */}
            <div className="flex flex-row gap-2">
              <div className="flex flex-col gap-2">
              <label className="text-xs text-gray-300">Load Flattened Image:</label>
              <textarea
                ref={flattenedImageInputRef}
                placeholder="Paste flattened image data (e.g., const int8_t input_data[784] = {...})"
                className="w-48 h-20 px-2 py-1 bg-gray-800 text-gray-200 border border-gray-600 rounded text-xs font-mono resize-none"
                onChange={(e) => {
                  const content = e.target.value;
                  if (content.trim()) {
                    try {
                      const parsedImage = parseFlattenedImage(content);
                      if (parsedImage) {
                        loadFlattenedImage(parsedImage);
                        setErrorMessage("");
                      } else {
                        setErrorMessage("Failed to parse image data. Please check the format.");
                      }
                    } catch (error) {
                      console.error('Error parsing image:', error);
                      setErrorMessage("Failed to parse image data. Please check the format.");
                    }
                  }
                }}
              />
              </div>
              <button
                onClick={() => {
                  const textarea = document.querySelector('textarea[placeholder*="flattened image"]') as HTMLTextAreaElement;
                  if (textarea) {
                    textarea.value = '';
                    setErrorMessage("");
                  }
                }}
                className="text-xs text-gray-400 hover:text-white px-2 py-1 rounded hover:bg-gray-700"
              >
                Clear
              </button>
              
              
            </div>
            <div className="flex items-center justify-center gap-2 mt-2">
            <button 
              onClick={() => {
                const arrayStr = pixelData
                  .map((v, _) => {
                    // Convert to 0-255 int for uint8_t
                    const intVal = Math.round(Math.max(0, Math.min(1, v)) * 255);
                    return intVal.toString();
                  })
                  .join(', ');

                const cArray = `const uint8_t input_data[${pixelData.length}] = { ${arrayStr} };`;

                // Update the textarea
                if (flattenedImageInputRef.current) {
                  flattenedImageInputRef.current.value = cArray;
                }
              }}
              className="px-4 py-2 bg-blue-600 text-white border border-blue-700 rounded hover:bg-blue-700 transition-colors font-medium"
            >
              Sync Canvas
            </button>
            </div>
          </div>
        </div>
        
        <div className="flex flex-col gap-2">
          <input
            type="file"
            accept=".json"
            onChange={handleWeightUpload}
            className="hidden"
            id="weight-upload"
          />

          {/* Quantized Model Input */}
          <div className="mt-4">
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm text-gray-300">Paste Quantized Model (C Header):</label>
              <button
                onClick={() => {
                  const textarea = document.getElementById('quantized-model-input') as HTMLTextAreaElement;
                  if (textarea) {
                    textarea.value = '';
                    setErrorMessage("");
                  }
                }}
                className="text-xs text-gray-400 hover:text-white px-2 py-1 rounded hover:bg-gray-700"
              >
                Clear
              </button>
            </div>
            <textarea
              id="quantized-model-input"
              placeholder="Paste your quantized model C header file content here..."
              className="w-full h-32 px-3 py-2 bg-gray-800 text-gray-200 border border-gray-600 rounded text-xs font-mono resize-none"
              onChange={(e) => {
                const content = e.target.value;
                if (content.trim()) {
                  try {
                    const parsedModel = parseQuantizedModel(content);
                    if (parsedModel && parsedModel.weights && 
                        parsedModel.weights.hidden1 && 
                        parsedModel.weights.hidden2 && 
                        parsedModel.weights.output) {
                      
                      // Update architecture
                      updateArchitecture(parsedModel.architecture);
                      
                      // Update network config
                      setNetworkConfig({
                        weights: parsedModel.weights,
                        classes: parsedModel.classes,
                        description: parsedModel.description
                      });
                      
                      // Add to available models
                      setAvailableModels((prevModels: AvailableModel[]) => {
                        const customModelExists = prevModels.some(model => model.name === 'Quantized Model');
                        if (!customModelExists) {
                          return [...prevModels, { name: 'Quantized Model', data: parsedModel }];
                        } else {
                          return prevModels.map(model => 
                            model.name === 'Quantized Model' ? { name: 'Quantized Model', data: parsedModel } : model
                          );
                        }
                      });
                      
                      setSelectedModel('Quantized Model');
                      setErrorMessage("");
                    } else {
                      setErrorMessage("Invalid model structure. Expected hidden1, hidden2, and output layers (hidden3 is optional).");
                    }
                  } catch (error) {
                    console.error('Error parsing model:', error);
                    setErrorMessage("Failed to parse quantized model. Please check the format.");
                  }
                }
              }}
            />
            <button
              onClick={() => {
                const sampleModel = `// Automatically generated header file
// Date: 2024-09-05 14:15:11.804110
// Quantized model exported from opt_Cosine_lr0.001_Aug_BitMnist_PerTensor_4bitsym_RMS_width16_16_0_bs128_epochs60.pth
// Generated by exportquant.py

#include <stdint.h>

#ifndef BITNETMCU_MODEL_H
#define BITNETMCU_MODEL_H

// Number of layers
#define NUM_LAYERS 3

// Maximum number of activations per layer
#define MAX_N_ACTIVATIONS 16

// Layer: L1
// QuantType: 4bitsym
#define L1_active
#define L1_bitperweight 4
#define L1_incoming_weights 784
#define L1_outgoing_weights 16
const uint32_t L1_weights[] = {
	0x11101121,0x109abccb,0xa9800111,0x10111101,0x121218ac,0xbca8a908,0x80000011,0x12122224,
	0x19dfc900,0x80008000,0x00811012,0x24431bff,0xfaa09000,0x81800001,0x10223431,0xaffffa88
};

// Layer: L2
// QuantType: 4bitsym
#define L2_active
#define L2_bitperweight 4
#define L2_incoming_weights 16
#define L2_outgoing_weights 16
const uint32_t L2_weights[] = {
	0x8110e2a1,0x1aa390f1,0x05cb0482,0x20800981,0x162013b8,0x804828b2,0x1b910933,0x200b518b
};

// Layer: L3
// QuantType: 4bitsym
#define L3_active
#define L3_bitperweight 4
#define L3_incoming_weights 16
#define L3_outgoing_weights 10
const uint32_t L3_weights[] = {
	0x9abbb84b,0xc92880bb,0xb1b28ca0,0x08828906,0x0b821919,0xaaa8993a,0xb38a1c8a,0x9912a028
};

#endif`;
                const textarea = document.getElementById('quantized-model-input') as HTMLTextAreaElement;
                if (textarea) {
                  textarea.value = sampleModel;
                  // Trigger the onChange event
                  const event = new Event('input', { bubbles: true });
                  textarea.dispatchEvent(event);
                }
              }}
              className="mt-2 w-full px-3 py-2 bg-gray-700 text-gray-300 border border-gray-600 rounded text-xs hover:bg-gray-600 transition-colors"
            >
              Load Sample MNIST Model
            </button>
            <button
              onClick={() => {
                const sampleModel3Hidden = `// Sample 3-hidden-layer model
// This demonstrates the 3 hidden layer functionality

#include <stdint.h>

#ifndef BITNETMCU_MODEL_H
#define BITNETMCU_MODEL_H

// Number of layers
#define NUM_LAYERS 4

// Maximum number of activations per layer
#define MAX_N_ACTIVATIONS 16

// Layer: L1 (Input -> Hidden1)
// QuantType: 4bitsym
#define L1_active
#define L1_bitperweight 4
#define L1_incoming_weights 784
#define L1_outgoing_weights 16
const uint32_t L1_weights[] = {
	0x11101121,0x109abccb,0xa9800111,0x10111101,0x121218ac,0xbca8a908,0x80000011,0x12122224,
	0x19dfc900,0x80008000,0x00811012,0x24431bff,0xfaa09000,0x81800001,0x10223431,0xaffffa88
};

// Layer: L2 (Hidden1 -> Hidden2)
// QuantType: 4bitsym
#define L2_active
#define L2_bitperweight 4
#define L2_incoming_weights 16
#define L2_outgoing_weights 16
const uint32_t L2_weights[] = {
	0x8110e2a1,0x1aa390f1,0x05cb0482,0x20800981,0x162013b8,0x804828b2,0x1b910933,0x200b518b
};

// Layer: L3 (Hidden2 -> Hidden3)
// QuantType: 4bitsym
#define L3_active
#define L3_bitperweight 4
#define L3_incoming_weights 16
#define L3_outgoing_weights 16
const uint32_t L3_weights[] = {
	0x9abbb84b,0xc92880bb,0xb1b28ca0,0x08828906,0x0b821919,0xaaa8993a,0xb38a1c8a,0x9912a028
};

// Layer: L4 (Hidden3 -> Output)
// QuantType: 4bitsym
#define L4_active
#define L4_bitperweight 4
#define L4_incoming_weights 16
#define L4_outgoing_weights 10
const uint32_t L4_weights[] = {
	0x12345678,0x9abcdef0,0x11111111,0x22222222,0x33333333,0x44444444,0x55555555,0x66666666
};

#endif`;
                const textarea = document.getElementById('quantized-model-input') as HTMLTextAreaElement;
                if (textarea) {
                  textarea.value = sampleModel3Hidden;
                  // Trigger the onChange event
                  const event = new Event('input', { bubbles: true });
                  textarea.dispatchEvent(event);
                }
              }}
              className="mt-2 w-full px-3 py-2 bg-gray-700 text-gray-300 border border-gray-600 rounded text-xs hover:bg-gray-600 transition-colors"
            >
              Load Sample 3-Hidden-Layer Model
            </button>
          </div>
          <style>{`
          select.custom-dropdown {
            text-align: center; /* Center the text */
            text-align-last: center; /* Center the selected text */
          }          
          select.custom-dropdown option {
            background-color: #000000; /* Change this to your desired color */
            color: #FFFFFFE0; /* Change this to your desired text color */
            text-align: center;
          }
        `}</style>

          <select
            value={selectedModel}
            onChange={(e) => {
              setSelectedModel(e.target.value);
              handleModelSelect(e.target.value);
            }}
            className="px-4 py-2 bg-gray-800  text-[#FFFFFF] border border-[#FFFFFF80] hover:bg-[#FFFFFF22] rounded custom-dropdown"
          >
            {availableModels.map(model => (
              <option key={model.name} value={model.name}>
                {model.name}
              </option>
            ))}
          </select>


          {networkConfig.description && (
            <div className="text-sm text-gray-300 min-w-[250px] mt-2 px-2 text-center">
            {networkConfig.description.split('\n').map((line, index) => (
              <Fragment key={index}>
                {line}
                <br />
              </Fragment>
            ))}
            </div>
          )}
          
          {/* Architecture Info */}
          <div className="text-xs text-gray-400 min-w-[250px] mt-2 px-2 text-center">
            <div>Input: {ARCHITECTURE.input} neurons</div>
            <div>Hidden1: {ARCHITECTURE.hidden1} neurons</div>
            <div>Hidden2: {ARCHITECTURE.hidden2} neurons</div>
            {ARCHITECTURE.hidden3 > 0 && <div>Hidden3: {ARCHITECTURE.hidden3} neurons</div>}
            <div>Output: {ARCHITECTURE.output} neurons</div>
            <div>Canvas: {canvasSize}×{canvasSize} grid</div>
            </div>
         
        </div>
      </div>
      
      <NetworkViz 
        activations={pixelData} 
        config={networkConfig}
        centerInput={centerInput}
        setCenterInput={setCenterInput}
      />
    </div>
  );
};

export default DrawingCanvas;